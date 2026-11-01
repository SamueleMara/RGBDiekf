#!/usr/bin/env python3
"""Summarize real completed/failed pilots without substituting preflight evidence.
Writes the seven requested reports and a bounded per-point/time numerical table.
Never initiates inference, calibration or GT evaluation.
"""
import argparse,ast,hashlib,json,subprocess,tempfile,sys,html,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io import json_write,serial

REPO=Path(__file__).resolve().parents[1]
DEFAULT=Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf')

def load(path,default=None):
 return json.loads(path.read_text()) if path.is_file() else default

def scan(parent):
 result=[];pointrows=[]
 for root in sorted(parent.glob('temporal_rgbd_next_*/*/*')):
  if not (root/'run_manifest.json').is_file():continue
  globalman=load(root/'run_manifest.json',{})
  for direction in ('forward','backward'):
   folder=root/f'free_{direction}';man=load(folder/'run_manifest.json')
   if not man:continue
   status=load(folder/'status.json',{'status':'INCOMPLETE_OR_RUNNING'})
   if status['status']=='INCOMPLETE_OR_RUNNING' and (root.parent/'INTERRUPTED_SUPERSEDED.json').exists():status=load(root.parent/'INTERRUPTED_SUPERSEDED.json')
   summary={'run':str(root),'pass':str(folder),'scene':globalman.get('arguments',{}).get('scene','unknown'),
     'mode':man.get('temporal_mode','legacy'),'direction':direction,'point_count':man.get('point_count',man.get('points_selected')),'frames_requested':len(man['processing_order']),
     'device':globalman.get('arguments',{}).get('device'),'status':status,'focal_control':man.get('diagnostic_focal_control',False),
     'manifest_mtime':(folder/'run_manifest.json').stat().st_mtime,'source_hashes':globalman.get('source_hashes',{}),
     'noise_assumption':man.get('noise_assumption'),'frames':[],'canonical_ID_mapping_verified':False}
   summary['viewer']=str(root/'debug/index.html') if summary['mode']=='legacy' else str(folder/'temporal/index.html')
   can=np.load(folder/'canonical.npz');sel=can['subsampled_to_full_index']
   summary['canonical_ID_mapping_verified']=bool(np.array_equal(can['point_id'],sel) and np.array_equal(can['original_source_ids'],can['full_endpoint_source'][sel]) and np.array_equal(can['canonical_xyz'],can['full_endpoint_xyz'][sel]))
   sm=load(folder/'subsample_manifest.json',{})
   summary['focal_manifest_old_selected_count_mismatch']=bool(summary['focal_control'] and sm.get('N_selected')!=len(sel))
   summary['actual_focal_reduction']=len(can['full_endpoint_xyz'])/len(sel)
   componentstats=[]
   for path in sorted((folder/'temporal').glob('frame_*.json')):
    d=load(path);hist=d['solver']['history'];summary['frames'].append({'frame':d['frame'],'rows':d['rows'],'eligible_points':d['observed_points'],
      'no_direct_row_points':d['prediction_only'],'assimilated_points':d['solver'].get('assimilated_points'),
      'converged':d['solver']['converged'],'accepted_iterations':sum(h['accepted'] for h in hist),'iterations':len(hist),
      'seconds':d.get('seconds_including_export',d['seconds']),'stage_times':d['stage_times'],'speed_median_m_s':d.get('speed_median_m_s'),
      'rgb_RMS':d.get('rgb_residual_rms'),'depth_RMS_m':d.get('depth_residual_rms_m'),'outside_foreground':d['rendered_outside_foreground']})
   if summary['mode']=='legacy':
    for path in sorted((folder/'numeric').glob('frame_*.json')):
     d=load(path);sensor=d['sensor'];update=d['filter_update'];processing=d['processing']
     summary['frames'].append({'frame':d['frame'],'rows':sensor['rows_accepted_final'],'eligible_points':sensor['unique_points_observed'],
      'no_direct_row_points':sensor['prediction_only_points'],'converged':update['not_converged_points']==0,
      'assimilated_points':None,'accepted_iterations':None,'iterations':processing.get('iekf_iterations'),
      'seconds':processing['wall_ms']/1000,'stage_times':processing.get('cuda_ms',{}),'scope':'legacy absolute RGB-D; different likelihood'})
   for p in (folder/'temporal').glob('components_*.json'):
    cc=load(p)['components'];dims=np.asarray([c['scalars'] for c in cc]);conds=[c['qr_condition'] for c in cc]
    componentstats.append({'frame':int(p.stem.split('_')[1]),'count':len(cc),'dimensions_median_p95_max':np.quantile(dims,[.5,.95,1]).tolist() if len(dims) else [],
      'QR_condition_max':max(conds,default=None),'constraint_rank_total':sum(c['constraint_rank'] for c in cc),
      'workspace_max_bytes':max((c['workspace_bytes'] for c in cc),default=0),'policy':'EXACT_WITHIN_COMPONENT; no linearized edge cuts; FEJ and finite lag are approximations'})
   summary['components']=componentstats
   summary['real_linearization_oracle']=load(root/'real_linearization_oracle_detailed.json') or load(root/'real_linearization_oracle.json')
   summary['real_schur_parity']=[load(p) for p in sorted((folder/'temporal').glob('real_schur_parity_*.json'))]
   gt=load(root/'evaluation_temporal_points_gt_next.json') or load(root/'evaluation_temporal_points_gt.json');summary['posthoc_GT']=gt.get('cloud_summaries',{}).get(direction) if gt else None
   for p in (folder/'temporal/traced_points').glob('*.json'):
    trace=load(p);original=trace['point_id']
    for row in trace['frames']:
     s=np.asarray(row['s_plus']);prior=np.asarray(row['s_minus']);P=np.asarray(row['P_plus']);raw=[]
     for entry in row['rows']:raw.append((row['frame'],entry['camera'],entry['current_pixel']))
     stat={'run':str(root),'mode':man['temporal_mode'],'scene':Path(summary['scene']).name,'focal_control':summary['focal_control'],'frame':row['frame'],
       'original_point_id':original,'local_array_index':row.get('local_array_index',original),'source':row['source'],
       'rows':len(row['rows']),'measurement_update_accepted':row.get('measurement_update_accepted'),'position_correction_m':float(np.linalg.norm(s[:3]-prior[:3])),
       'velocity_correction_m_s':float(np.linalg.norm(s[3:]-prior[3:])),'speed_m_s':float(np.linalg.norm(s[3:])),
       'P_position_trace_m2':float(np.trace(P[:3,:3])),'P_min_eigenvalue_mixed_units':float(np.linalg.eigvalsh(P).min()),
       'camera_reason':row['camera_reason'],'reference_frames':row['reference_frames'],'converged':row['solver']['converged'],
       'raw_current_keys':raw,'trace':str(p),'references':[{'camera':e['camera'],'reference_frame':e['reference_frame'],'reference_pixels':e['reference_pixels'],
          'residual':e['residual'],'q':e['q'],'cross_time_norm':float(np.linalg.norm(e['Cov_current_reference'])) if e.get('Cov_current_reference') is not None else None} for e in row['rows']]}
     if gt:
      match=[e for e in gt['points'] if e['original_point_id']==original and e['frame']==row['frame'] and e['direction']==direction]
      stat['posthoc_GT']=match[0] if match else None
     pointrows.append(stat)
   result.append(summary)
 return result,pointrows


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results-parent',type=Path,default=DEFAULT);ap.add_argument('--output',type=Path)
 ap.add_argument('--calibration-analysis',type=Path,required=True);ap.add_argument('--regressions',type=Path,required=True);ap.add_argument('--suite-log',type=Path,required=True)
 a=ap.parse_args();out=a.output or Path(tempfile.mkdtemp(prefix='temporal_rgbd_next_validation_',dir=a.results_parent));out.mkdir(parents=True,exist_ok=True)
 runs,points=scan(a.results_parent);tests=load(REPO/'tests/temporal_test_results.json',{});cal=load(a.calibration_analysis,{});reg=load(a.regressions,{})
 groups={};comparisons=[]
 for r in runs:
  key=(r['scene'],r['focal_control'],r['frames_requested'],r['direction']);g=groups.setdefault(key,{})
  if r['mode'] not in g or r['manifest_mtime']>g[r['mode']]['manifest_mtime']:g[r['mode']]=r
 for key,g in groups.items():
  if 'full' not in g or 'simplified' not in g:continue
  f,s=g['full'],g['simplified'];fc=np.load(Path(f['pass'])/'canonical.npz');sc=np.load(Path(s['pass'])/'canonical.npz')
  record={'scene':Path(key[0]).name,'focal':key[1],'frames':key[2],'direction':key[3],'full_run':f['run'],'simplified_run':s['run'],
   'identical_initial_mean':bool(np.array_equal(fc['canonical_xyz'],sc['canonical_xyz'])),
   'identical_initial_source_covariance':bool(np.array_equal(fc['canonical_source_covariance'],sc['canonical_source_covariance'])),
   'identical_original_source_pixels':bool(np.array_equal(fc['original_source_ids'],sc['original_source_ids'])),
   'full_status':f['status']['status'],'simplified_status':s['status']['status'],'point_comparisons':[]}
  if 'legacy' in g:
   lc=np.load(Path(g['legacy']['pass'])/'canonical.npz')
   record['legacy_control']={'run':g['legacy']['run'],'status':g['legacy']['status']['status'],
       'identical_initial_mean':bool(np.array_equal(lc['canonical_xyz'],sc['canonical_xyz'])),
       'identical_initial_source_covariance':bool(np.array_equal(lc['canonical_source_covariance'],sc['canonical_source_covariance'])),
       'identical_original_source_pixels':bool(np.array_equal(lc['original_source_ids'],sc['original_source_ids'])),
       'likelihood_scope':'unchanged legacy absolute RGB-D; do not compare raw objectives with temporal warp objectives'}
  fp={(r['frame'],r['original_point_id']):r for r in points if r['run']==f['run']};sp={(r['frame'],r['original_point_id']):r for r in points if r['run']==s['run']}
  for pk in sorted(fp.keys()&sp.keys()):
   left,right=fp[pk],sp[pk];lk={tuple(k) for k in left['raw_current_keys']};rk={tuple(k) for k in right['raw_current_keys']}
   record['point_comparisons'].append({'frame':pk[0],'original_point_id':pk[1],'shared_current_raw_pixels':len(lk&rk),'full_pixels':len(lk),'simplified_pixels':len(rk),
    'full_Pxx_trace':left['P_position_trace_m2'],'simplified_Pxx_trace':right['P_position_trace_m2'],'full_speed':left['speed_m_s'],'simplified_speed':right['speed_m_s'],
    'full_update_accepted':left['measurement_update_accepted'],'simplified_update_accepted':right['measurement_update_accepted']})
  comparisons.append(record)
 suite=a.suite_log.read_text() if a.suite_log.exists() else '';finished='\nRan ' in suite
 import re
 suite_summary=re.findall(r'Ran (\d+) tests in ([\d.]+)s',suite)
 evidence={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'dirty_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True).splitlines(),
  'runs':runs,'point_rows':points,'full_simplified_comparisons':comparisons,'tests':tests,'calibration':cal,'regressions':reg,'broad_suite_finished':finished,'broad_suite_summary':suite_summary[-1] if suite_summary else None,
  'broad_suite_failure_names':re.findall(r'^FAIL: ([^\n]+)',suite,re.M),
  'broad_suite_error_names':re.findall(r'^ERROR: ([^\n]+)',suite,re.M),
  'integrity_rerun_log':'/tmp/temporal_next_integrity_rerun.log','production_validation':False}
 json_write(out/'evidence.json',serial(evidence))
 edited=['train.py','estimator/temporal_noise.py','estimator/temporal_full_graph.py','estimator/temporal_simplified_iekf.py','estimator/temporal_run.py',
  'estimator/temporal_zbuffer.py','estimator/temporal_rgbd_terms.py','debug/temporal_debug.py','tools/run_temporal_rgbd_critical.py',
  'tools/calibrate_temporal_warp_noise.py','tools/audit_temporal_regressions.py','tools/audit_temporal_saved_linearizations.py','tools/evaluate_temporal_points_gt.py','tools/qa_temporal_debug.py','tools/report_temporal_next_validation.py']
 mapping='# Actual implementation map\n\n'+evidence['git_commit']+'; dirty working tree preserved.\n\nCLI train.parse/run_scene → Dataset → build_catalogue + original source IDs → first_hit_pixel_owner → ReferenceTable → build_temporal_rows → constrained full QR/Schur OR independent simplified IEKF → arrays, numeric traces and HTML.\n\n'
 for name in edited:
  p=REPO/name;tree=ast.parse(p.read_text());symbols=[f'{n.name}:{n.lineno}-{n.end_lineno}' for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
  mapping+=f'- `{name}`: '+', '.join(symbols)+f'; SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}\n'
 (out/'IMPLEMENTATION_MAP.md').write_text(mapping)
 noise='''# Noise semantics and calibration

**IDEAL_WARP_ASSUMPTION_UNCALIBRATED** sets only additional conditional R_warp_model=0. It is an exact equality in full mode, not zero total measurement noise. Raw current/reference sensor observations appear once, source depth is coupled to its point, and all retained state/pixel cross blocks participate. Simplified uses R_current+sum(w² R_ref) and discards cross-time/cross-point correlations.

Sensor noise is the existing documented guess or conditional same-camera temporal MAD/quantization estimate, with provenance; not independent laboratory calibration. The temporal path neither computes nor uses cross-camera model mismatch. Units: encoded RGB [0,1], depth metres; covariances have products of those units. Q uses signed actual time differences and approved white-acceleration density. Config/measured warp files stay SPD. Ideal zero is PSD and accepted only as the separate discrepancy term. Prior-predictive full innovation must remain SPD.

Version 1 covariance_rgbd already represented ADDITIONAL discrepancy; version 2 records that semantics explicitly, preserving old interpretation. Exact ideal constraints are solved in their nullspace, and constrained Schur retains both Gaussian FEJ rows and remaining equalities. No jitter or stochastic information eigenvalue truncation.

## Measured study

'''
 noise+=f"Status: **{cal.get('status')}**. Candidate counts: {cal.get('candidate_counts')}. Protocol: {cal.get('held_out_protocol')}. Excluded boundary pairs: {cal.get('excluded_shared_boundary_pairs')}.\n\n"
 noise+='The primary study estimates sensor laws separately for each of the eight cameras using calibration-block frames only. Those conditional static-majority MAD estimates do not use held-out frames. Pilot inference retains its existing pooled per-scene sensor estimate on the selected prefix; it is not independently held-out calibration. Neither the separate calibration study nor posthoc GT tunes pilot parameters. The earlier documented-sensor-guess study is preserved separately for sensitivity comparison.\n\n'
 for reason in cal.get('not_identifiable_reasons',[]):noise+='- '+reason+'\n'
 for split,d in cal.get('empirical_temporal_difference',{}).items():
  noise+=f'\n{split}: weighted count {d["effective_samples"]}; empirical-minus-sensor eigenvalues {d["empirical_minus_sensor_eigenvalues_diagnostic_only"]}. No clipping/PSD projection applied.\n'
 noise+='\nFull matrices, correlations, camera/depth/gradient/edge/warp/forward-backward breakdowns and code/data hashes are in calibration_analysis.json. Optical-flow-selected residuals are conditional candidates, not independent known correspondences. No usable calibrated R_warp was produced.\n'
 noise+='\nCameras/timing and noise/process hyperparameters are conditioned on their supplied/estimated values; calibration-parameter uncertainty is not added independently per point. Sensor and warp terms are distinct. Current/reference state covariance and cross covariance are carried in the full joint graph, never added again to R_warp.\n'
 (out/'NOISE_SEMANTICS_AND_CALIBRATION.md').write_text(noise);json_write(out/'calibration_analysis.json',cal)
 text='# Individual classification of seven broad-suite failures\n\nCaptured pre-edit dirty tracked source baseline reconstructed in /tmp; active source was never checked out/reset. See regressions.json for baseline hashes, exact assertions and original log paths.\n\n'
 for c in reg.get('cases',[]):text+=f'- `{c["case"]}`: **{c["classification"]}**; baseline {c["runs"]["baseline"]["returncode"]}, active {c["runs"]["active"]["returncode"]}.\n'
 text+=f'\nNew full-suite completed: {finished}; summary {suite_summary[-1] if suite_summary else "still running"}.\n'
 text+='\nThe broad run also caught one protected-source integrity error because the warp schema was edited concurrently with the archived audit. That exact test was rerun with source edits paused: 1 test passed in 60.993s (see logs/integrity_rerun.log). Classified ENVIRONMENT_CONCURRENT_SOURCE_EDIT, not silently erased from the 326-test log. The full suite is not reported green.\n'
 text+='\nEarly focal controls retained endpoint-only N_selected in the subsample manifest. Actual canonical point_id/source mappings remained correct; the selection manifest now reports its true focal count/reduction, and historical mismatches are flagged in evidence.json rather than silently rewritten.\n\nNew implementation bugs found by actual runs: cuSOLVER batch/workspace failures (bounded independent diagnostics); whole-cloud acceptance blocking independent updates (per-point/per-true-component backtracking and common-domain verification); never-accepted simplified updates incorrectly contracting covariance (propagated covariance retained); ideal-cell refresh requiring clean-pixel reprojection (exact constraint reprojected after rebind). These were not dismissed as legacy failures.\n'
 (out/'REGRESSION_7_FAILURES.md').write_text(text);json_write(out/'regressions.json',reg)
 text=f'# Temporal oracle tests\n\nLatest focused suite: {tests.get("tests_run")} tests, {tests.get("failures")} failures, {tests.get("errors")} errors, {tests.get("skips")} skips.\n\n'
 text+='Independent dense KKT oracle versus nullspace QR, Schur state/pixel cross-covariance, and repeated-reference variance-floor witnesses:\n\n```json\n'+json.dumps(tests.get('ideal_warp_oracle_witnesses'),indent=2)+'\n```\n\n'
 text+='Other finite-difference, shared raw-pixel, CPU/CUDA, legacy parity and physical toy-motion results are in tests.json. Passing numerical fixtures does not establish real scene accuracy or calibration.\n'
 for r in runs:
  oracle=r.get('real_linearization_oracle')
  if oracle:
   cc=oracle['components'];gg=oracle['geometry'];ww=oracle['warp_operator']
   text+=f'\n## Actual saved operator: {Path(r["run"]).name}\n\n{len(cc)} dense SciPy component replays; max covariance discrepancy {max((c["dense_SciPy_vs_saved_Torch_covariance_max_abs"] for c in cc),default=None)}; max relative discrepancy {max((c["covariance_relative_max"] for c in cc),default=None)}. CPU/CUDA owner-record disagreements {sum(g["selected_record_disagreements"] for g in gg)}; warp/J maximum absolute discrepancy {max((max(w["h_max_abs"],w["H_current_max_abs"],w["H_reference_max_abs"]) for w in ww),default=None)}.\n'
   text+=f'\nNext-linearization Newton step discrepancy (not the accepted trajectory): max absolute {max((c["step_max_abs"] for c in cc),default=None)}, max relative {max((c.get("step_relative_max",0) for c in cc),default=None)}. Actual conditioning and normalized stationarity are reported; no blanket 1e-13 mean-accuracy claim on the ill-conditioned real operators.\n'
  schur=[p for group in r['real_schur_parity'] if group for p in group]
  if schur:text+=f'\nActual lag slides in {Path(r["run"]).name}: {len(schur)} point checks; max current covariance discrepancy {max(p["state_covariance_max_abs"] for p in schur)}, max state-pixel cross discrepancy {max(p["cross_covariance_max_abs"] for p in schur)}.\n'
 (out/'TEMPORAL_ORACLE_TEST_REPORT.md').write_text(text);json_write(out/'tests.json',tests)
 table=['# Actual critical pilots','','Eligible rows, accepted updates and convergence are separate. Focal controls remove other points from the occlusion catalogue and are GLOBAL_APPROXIMATION controls, not full-cloud validation. No preflight is counted.','',
 '| Scene | Mode | Device | Points | Requested frames | Saved filtered frames | Focal | Status | Converged frames |','|---|---|---|---:|---:|---:|---|---|---:|']
 for r in runs:table.append(f'| {Path(r["scene"]).name} | {r["mode"]} | {r["device"]} | {r["point_count"]} | {r["frames_requested"]} | {len(r["frames"])} | {r["focal_control"]} | {r["status"]["status"]} | {sum(f["converged"] for f in r["frames"])} |')
 table+=['','## Original-ID point/time evidence','', '| Scene | Mode | Focal | Original ID | Frame | Eligible rows | Update accepted | Position correction m | Speed m/s | tr(Pxx) m² | Converged |','|---|---|---|---:|---:|---:|---|---:|---:|---:|---|']
 for r in points:table.append(f'| {r["scene"]} | {r["mode"]} | {r["focal_control"]} | {r["original_point_id"]} | {r["frame"]} | {r["rows"]} | {r["measurement_update_accepted"]} | {r["position_correction_m"]:.6g} | {r["speed_m_s"]:.6g} | {r["P_position_trace_m2"]:.6g} | {r["converged"]} |')
 table+=['','Every source/ownership/reference/innovation/cross-covariance record links to its original trace in evidence.json. An eligible-but-rejected row is not successful tracking. Posthoc GT errors, where computed, are in evidence.json and each run/evaluation_temporal_points_gt.json; they never enter inference.']
 table+=['','Full/simplified matched-initialization checks and per-point raw-pixel intersection counts are in evidence.json/full_simplified_comparisons. The image sequences/cameras are identical; final selected rows may differ because estimated states/ownership differ. No silent restriction to a favorable intersection is used in inference.']
 table+=['','## Full-cloud frame-10 comparison with unchanged legacy control','', '| Scene | Mode | Points | Median static error m | Median moving error m | p95 moving error m |','|---|---|---:|---:|---:|---:|']
 for r in runs:
  if not r['focal_control'] and r['frames_requested']==11 and r['posthoc_GT']:
   metric=r['posthoc_GT']['frames'][-1]
   table.append(f'| {Path(r["scene"]).name} | {r["mode"]} | {r["point_count"]} | {metric["static_error_median_m"]} | {metric["moving_error_median_m"]} | {metric["moving_error_p95_m"]} |')
 table+=['','The legacy control uses the original absolute projective RGB-D likelihood and existing cross-camera mismatch model. Initial means, source covariances and raw-source identities are compared explicitly; different likelihood objectives are not numerically ranked. A zero static median can include unchanged/prediction-only points and does not establish successful tracking.']
 table+=['','## Post-inference GT diagnostics, completed 11-frame runs','', '| Scene/run | Point ID | True motion m | Estimated motion m | Position error m | Position NEES | Max true reference flow px |','|---|---:|---:|---:|---:|---:|---:|']
 for p in points:
  gt=p.get('posthoc_GT')
  if p['frame']==10 and gt:table.append(f'| {Path(p["run"]).name} | {p["original_point_id"]} | {gt.get("true_displacement_from_anchor_m")} | {gt.get("estimated_displacement_from_anchor_m")} | {gt["position_error_m"]} | {gt["position_NEES"]} | {gt.get("true_current_reference_pixel_motion_max")} |')
 table+=['','GT is applied to the measured canonical anchor using supplied joint motion and nearest GT-cloud labels. This is a conditional motion-error/NEES diagnostic, not an independent noisy-anchor calibration experiment. Conservative common-owner/raw-pixel/reference domains can stall: sparse numerical correctness does not resolve physical correspondence or large image motion. No velocity clamp, arbitrary association threshold or GT label was added to make errors smaller.']
 (out/'CRITICAL_PILOT_RESULTS.md').write_text('\n'.join(table)+'\n')
 perf='# Performance and component growth\n\nNo graph edge is split; oversized or time-expired components/runs fail explicitly. Full covariance is a constrained joint marginal, simplified covariance is an independent approximation. Focal selection is explicitly a reduced occlusion model. Times from concurrent runs are contention-affected, not isolated throughput benchmarks.\n\n'
 for r in runs:
  perf+=f'## {Path(r["run"]).name} ({r["device"]}, {r["point_count"]} points)\n\nStatus {r["status"]}.\n\n'
  if r['frames']:perf+='Per-frame total seconds: '+str([round(f['seconds'],3) for f in r['frames']])+'\n\n'
  if r['components']:perf+='Component profiles: `'+str(r['components'])+'`\n\n'
 perf+='Detailed input/projection-sort/interpolation/assembly/solve/Schur/covariance/diagnostic/viewer/reference-refresh times and CUDA high-water bytes remain in each pass temporal/timing_memory.csv. Sampling and sorting overlap solve wall time. A failed initialization has no solved-component profile and is not substituted with a successful result.\n'
 (out/'PERFORMANCE_AND_COMPONENT_GROWTH.md').write_text(perf)
 commands='''# Verified operator commands

These arguments were exercised on bounded runs. NO 60-frame or ten-scene production workload was launched. Full-mode production remains unvalidated/infeasible under the recorded budgets; ideal is always uncalibrated. Nonconvergent/failed pilots do not authorize a scientific success claim.

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
```

Three-frame full-cloud diagnostic (full can return FULL_RUNTIME_LIMIT):

```bash
bash tools/run_temporal_rgbd_critical.sh --mode full,simplified --direction forward \\
  --frame-limit 3 --scenes storage_45135,USB_100109 --temporal-warp-noise ideal \\
  --temporal-runtime-budget-seconds 120
```

Eleven-frame focal controls (explicitly reduced occlusion catalogue, exact original IDs):

```bash
bash tools/run_temporal_rgbd_critical.sh --mode full,simplified --direction forward \\
  --frame-limit 11 --scenes storage_45135,USB_100109,blade_103706,stapler_103111 \\
  --temporal-warp-noise ideal --focal-controls --device cpu \\
  --temporal-runtime-budget-seconds 900
```

Full 60-frame critical-scene command, FOR HUMAN USE AFTER reviewing failures:

```bash
bash tools/run_temporal_rgbd_critical.sh --mode full,simplified --direction both \\
  --scenes storage_45135,USB_100109 --temporal-warp-noise ideal \\
  --temporal-runtime-budget-seconds 120
```

All ten scenes, all 60 frames/eight cameras/both directions (not production-cleared):

```bash
bash tools/run_temporal_rgbd_critical.sh --mode full,simplified --direction both \\
  --temporal-warp-noise ideal --temporal-runtime-budget-seconds 120
```

No frame-limit means all frames; B1 anchors at original frame 59. The explicit budget is a computational guard, not a physical parameter. It is expected to fail on full-cloud full mode with the current backend; removing it does not establish feasibility. No calibrated command is given because calibration did not identify a justified R_warp. Data/output roots default to the user's actual synthetic folder and SSD. Each invocation creates a unique child; no source snapshots. Live actual estimation: SCENE_MODE/free_DIRECTION/temporal/index.html. Completed full viewer: SCENE_MODE/debug/index.html.
'''
 (out/'RUN_COMMANDS.md').write_text(commands)
 summary=f'''# Temporal RGB-D next-actions validation

**Production validation: NOT SATISFIED.** This report contains actual filtering, not preflight-only rendering.

Evidence folder: `{out}`.

Focused CPU/CUDA suite: {tests.get('tests_run')} tests; {tests.get('failures')} failures, {tests.get('errors')} errors, {tests.get('skips')} skips. Broad suite finished: {finished}. Full ideal discrepancy uses exact constraints with sensor noise active; calibration study: {cal.get('status')}.

'''
 summary+='\n'.join('- ['+name+'](<'+str(out/name)+'>)' for name in ['IMPLEMENTATION_MAP.md','NOISE_SEMANTICS_AND_CALIBRATION.md','REGRESSION_7_FAILURES.md','TEMPORAL_ORACLE_TEST_REPORT.md','CRITICAL_PILOT_RESULTS.md','PERFORMANCE_AND_COMPONENT_GROWTH.md','RUN_COMMANDS.md'])+'\n\n'
 summary+='Actual estimated HTML pages (inspect status/convergence and focal qualification):\n\n'
 for r in runs:
  if r['frames']:summary+=f'- [{Path(r["run"]).name}, {len(r["frames"])} saved frames](<{r["viewer"]}>) — {r["status"]["status"]}, focal={r["focal_control"]}.\n'
 summary+='\nDepth sort is not physical correspondence. Conservative hard-owner/reference domains can stall at boundaries. Local translational warp is not valid for all rotation/occlusion; finite-lag FEJ covariance is conditional. Successful files/eligible rows are not successful tracking, and failed or incomplete prefixes are not complete pilots. No claim of calibrated uncertainty or full-scene production readiness is made.\n'
 (out/'README.md').write_text(summary);(REPO/'reports/temporal_rgbd_next_validation.md').write_text(summary);json_write(REPO/'reports/temporal_rgbd_next_validation.json',serial(evidence))
 rows=[]
 for r in runs:
  link=os.path.relpath(Path(r['viewer']),out)
  rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [Path(r['scene']).name,r['mode'],r['device'],r['point_count'],r['frames_requested'],r['focal_control'],r['status']['status'],sum(f['converged'] for f in r['frames'])])+f'<td><a href="{html.escape(link,quote=True)}">Actual diagnostics</a></td></tr>')
 pairs=[]
 for c in comparisons:
  if c['frames']==11 and c['focal'] and c['full_status']==c['simplified_status']=='COMPLETE':pairs.append({'name':c['scene']+' — reduced focal cloud, 11 frames','full':os.path.relpath(Path(c['full_run'])/'free_forward/temporal/index.html',out),'simplified':os.path.relpath(Path(c['simplified_run'])/'free_forward/temporal/index.html',out)})
 page='''<!doctype html><meta charset="utf-8"><title>Actual temporal RGB-D validation</title><style>body{background:#101723;color:#e5edf9;font:16px system-ui;margin:24px}a{color:#6ad1ff}.warning{background:#562d24;padding:18px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #42516b;padding:6px}iframe{width:49%;height:1000px;border:1px solid #42516b}select{padding:10px}</style><h1>Actual temporal RGB-D filtering: scientific acceptance failed</h1><p class="warning">IDEAL_WARP_ASSUMPTION_UNCALIBRATED. Completed files are not calibrated or accurate tracking. Full-cloud full-mode runs hit runtime limits. Focal controls remove other occluders and are explicitly reduced-cloud diagnostics. No preflight is counted as estimation.</p><p><a href="README.md">Report index</a> · <a href="CRITICAL_PILOT_RESULTS.md">Errors and original-ID traces</a> · <a href="NOISE_SEMANTICS_AND_CALIBRATION.md">Noise / identifiability</a> · <a href="TEMPORAL_ORACLE_TEST_REPORT.md">Independent covariance oracles</a> · <a href="RUN_COMMANDS.md">Commands and limits</a></p><h2>Same-initialization focal comparison</h2><select id="pair"></select><p>Left FULL; right SIMPLIFIED. Both use actual measured RGB-D. Inspect accepted updates, convergence, point trajectories and uncertainty, not just eligible row counts.</p><iframe id="full"></iframe> <iframe id="simple"></iframe><h2>All recorded attempts</h2><table><tr><th>Scene</th><th>Mode</th><th>Device</th><th>Points</th><th>Frames</th><th>Focal</th><th>Execution status</th><th>Converged frames</th><th>View</th></tr>'''+''.join(rows)+'''</table><script>const pairs='''+json.dumps(pairs).replace('<','\\u003c')+''';const select=document.getElementById('pair');pairs.forEach((p,i)=>select.add(new Option(p.name,i)));function show(){let p=pairs[+select.value];if(p){document.getElementById('full').src=p.full;document.getElementById('simple').src=p.simplified}}select.onchange=show;show();</script>'''
 (out/'index.html').write_text(page)
 cards='<h2>Actual full-cloud estimates: simplified covariance, uncalibrated</h2><p>All available source points and eight cameras; 11 frames. These are estimated trajectories, but tracking/scientific acceptance failed. Open the 3D geometry, measured-image panels or per-point evolution directly.</p>'
 for r in runs:
  if not r['focal_control'] and r['mode']=='simplified' and r['frames_requested']==11 and r['status']['status']=='COMPLETE':
   link=html.escape(os.path.relpath(Path(r['pass'])/'temporal/index.html',out),quote=True)
   cards+=f'<p><b>{html.escape(Path(r["scene"]).name)}</b> — {r["point_count"]:,} points: <a href="{link}#viewport">Actual 3D evolution</a> · <a href="{link}#panel">RGB-D / warp / residual panels</a> · <a href="{link}#pointposition">Original-point trajectories and covariance</a> · <a href="{link}#objective">Convergence</a></p>'
 page=page.replace('<h2>Same-initialization focal comparison</h2>',cards+'<h2>Same-initialization focal comparison</h2>')
 page=page.replace('IDEAL_WARP_ASSUMPTION_UNCALIBRATED. Completed files','New temporal modes: IDEAL_WARP_ASSUMPTION_UNCALIBRATED. Legacy controls retain their original absolute RGB-D likelihood. Completed files')
 (out/'index.html').write_text(page)
 print(out)
if __name__=='__main__':main()
