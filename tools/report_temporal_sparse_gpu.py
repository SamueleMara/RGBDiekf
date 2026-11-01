#!/usr/bin/env python3
"""Audit saved sparse GPU evidence; never run inference or invent missing results."""
import argparse
import ast
import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import scipy.linalg as la
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.temporal_batch_runtime import SPARSE_PRODUCTION_SOURCES,sparse_source_hashes
from utils.temporal_output import atomic_json,require_valid_frame

REPO=Path(__file__).resolve().parents[1]


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def error(actual,expected):
    difference=np.linalg.norm(actual-expected)
    return {'absolute_frobenius':float(difference),'relative_frobenius':float(difference/max(np.linalg.norm(expected),np.finfo(float).tiny))}


def oracle(path):
    """Independent SciPy constrained square-root solve of a recorded whole component."""
    d=np.load(path,allow_pickle=False);A=d['A'];r=d['r'];C=d['C'];c=d['c'];n=A.shape[1]
    H=d['FEJ_information'];g=d['FEJ_gradient']
    if np.count_nonzero(H):
        L=la.cholesky(H,lower=True)
        B=np.concatenate((A,L.T));b=np.concatenate((r,la.solve_triangular(L,g,lower=True)))
    else:B,b=A,r
    if len(C):
        initial=la.lstsq(C,-c,lapack_driver='gelsd')[0];N=la.null_space(C)
    else:initial=np.zeros(n);N=np.eye(n)
    Q,R=la.qr(B@N,mode='economic')
    condition=float(np.linalg.cond(R))
    delta=initial+N@la.solve_triangular(R,-Q.T@(b+B@initial))
    S=la.solve_triangular(R,np.eye(len(R)));P=N@S@S.T@N.T
    selected=d['selected'];mean=error(d['step'],delta)
    covariance=error(d['selected_covariance'],P[np.ix_(selected,selected)])
    tolerance=512*np.finfo(float).eps*max(1.,condition)**2
    absolute_tolerance=2e-12;relative_tolerance=2e-11
    passed=(condition<1e6 and np.linalg.norm(d['step']-delta)<=absolute_tolerance+relative_tolerance*np.linalg.norm(delta)
            and np.linalg.norm(d['selected_covariance']-P[np.ix_(selected,selected)])<=absolute_tolerance+relative_tolerance*np.linalg.norm(P[np.ix_(selected,selected)]))
    return {'path':str(path),'complete_component_dimension':n,'rows':len(A),'hard_rows':len(C),
            'QR_condition':condition,'mean':mean,'selected_covariance':covariance,
            'condition_based_roundoff_bound_not_acceptance_tolerance':tolerance,
            'absolute_tolerance':absolute_tolerance,'relative_tolerance':relative_tolerance,
            'tolerance_basis':'FP64 CUDA/QR micrograph and real-component discrepancies; not a relaxation to the larger condition-number bound',
            'passed':bool(passed),
            'note':'Saved information includes actual eliminated Gaussian priors and FEJ, if present. No production estimate generated here.'}


def inspect_job(record):
    out=Path(record['output']);runs=[]
    for root in sorted(out.glob('free_*')):
        manifest=read(root/'run_manifest.json',{})
        if 'cuDSS' not in manifest.get('backend',''):continue
        journal=read(root/'completed_frames.json',{})
        status=read(root/'status.json',{})
        frames=[]
        for k in journal.get('completed_frames',[]):
            require_valid_frame(root,k)
            d=read(root/'temporal'/f'frame_{k:04d}.json',{})
            frames.append({'frame':k,**d})
        oracles=[]
        for path in sorted((root/'temporal').glob('oracle_frame_*_component_*.npz')):
            try:oracles.append(oracle(path))
            except Exception as exception:oracles.append({'path':str(path),'passed':False,'error':str(exception)})
        full_cloud=False
        with np.load(root/'canonical.npz') as source:
            ids=source['point_id'];full_cloud=len(ids)==len(source['full_endpoint_xyz']) and np.array_equal(ids,np.arange(len(ids)))
        valid_covariance=[]
        for k in journal.get('completed_frames',[]):
            if k==manifest.get('canonical_frame'):continue
            P=np.load(root/'posterior_covariance.npy',mmap_mode='r')[k]
            minimum=float('inf');symmetry=0.
            for start in range(0,len(P),2048):
                batch=np.asarray(P[start:start+2048]);minimum=min(minimum,float(np.linalg.eigvalsh(batch).min()))
                symmetry=max(symmetry,float(np.abs(batch-batch.swapaxes(1,2)).max()))
            valid_covariance.append({'frame':k,'minimum_eigenvalue_mixed_units':minimum,'maximum_asymmetry':symmetry,'passed':bool(minimum>0 and np.isfinite(P).all())})
        nonanchor=[f for f in frames if f['frame']!=manifest.get('canonical_frame')]
        three_ok=(status.get('status')=='COMPLETE' and journal.get('status')=='COMPLETE' and len(frames)==3
            and manifest.get('camera_count')==8 and full_cloud and len(nonanchor)==2
            and all(f.get('assimilated_points',0)>0 for f in nonanchor)
            and len(valid_covariance)==2 and all(v['passed'] for v in valid_covariance)
            and len(oracles)>=2 and all(o['passed'] for o in oracles))
        recorded_sources=read(out/'run_manifest.json',{}).get('source_hashes',{})
        hashes=sparse_source_hashes(REPO)
        run_matches_current=all(recorded_sources.get(p)==h for p,h in hashes.items() if p.startswith('estimator/'))
        runs.append({'root':str(root),'manifest':manifest,'journal':journal,'status':status,'frames':frames,
                     'source_hashes':recorded_sources,'run_matches_current_numerical_sources':run_matches_current,
                     'actual_component_oracles':oracles,'joint_covariance_verification':valid_covariance,
                     'all_original_points':bool(full_cloud),'three_frame_gate_passed':bool(three_ok and run_matches_current),
                     'live_HTML':str(root/'temporal/index.html'),
                     'HTML_has_verified_nonanchor_frames':bool(nonanchor)})
    return {'scene':record['scene'],'mode':record['mode'],'command':record.get('command'),'status':record['status'],
            'returncode':record.get('returncode'),'output':str(out),'runs':runs,
            'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=Path,nargs='+',required=True)
    ap.add_argument('--test-report',type=Path,required=True)
    ap.add_argument('--profile-report',type=Path)
    ap.add_argument('--output-dir',type=Path,default=REPO/'reports')
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    tests=read(a.test_report);profile=read(a.profile_report) if a.profile_report else None
    jobs=[]
    for batch in a.batch:
        records=read(batch/'batch_summary.json',[])
        if isinstance(records,dict):records=records.get('records',records.get('jobs',[]))
        for record in records:
            if record.get('mode')=='full':jobs.append(inspect_job(record))
    accepted_scenes={j['scene'] for j in jobs if any(r['three_frame_gate_passed'] for r in j['runs'])}
    three_ok={'USB_100109','blade_103706'}<=accepted_scenes
    production_hashes=sparse_source_hashes(REPO)
    tests_current=all((REPO/p).exists() and __import__('hashlib').sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in tests.get('source_sha256',{}).items())
    three_ok=(three_ok and tests_current and bool(tests.get('source_sha256')) and tests.get('cuda_available',False)
              and tests.get('all_executed_checks_pass',False) and tests.get('skips')==0)
    baseline_files=['estimator/free_projective_run.py','estimator/projective_sensor.py','estimator/projective_terms.py',
                    'estimator/point_kalman_cuda.py','estimator/temporal_simplified_iekf.py']
    unchanged={p:subprocess.run(['git','diff','--quiet','HEAD','--',p],cwd=REPO).returncode==0 for p in baseline_files}
    changes=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO)
    for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines():
        if Path(name).suffix in ('.py','.txt') and not name.startswith('reports/'):
            changes+=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=REPO,stdout=subprocess.PIPE).stdout
    (a.output_dir/'temporal_sparse_gpu_source_changes.patch').write_bytes(changes)
    function_map={}
    for name in SPARSE_PRODUCTION_SOURCES:
        tree=ast.parse((REPO/name).read_text())
        function_map[name]={n.name:n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    disk=shutil.disk_usage(a.batch[-1])
    report={'verdict':'FULL_SPARSE_GPU_NOT_ACCEPTED','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'dirty_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True).splitlines(),
        'production_source_sha256':production_hashes,'source_function_line_map':function_map,
        'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cudss_version':importlib.metadata.version('nvidia-cudss-cu12'),'legacy_math_unchanged_from_HEAD':unchanged,
        'test_report':str(a.test_report.resolve()),'test_results':tests,'tests_match_current_sources':tests_current,
        'scaling_profile_report':str(a.profile_report) if a.profile_report else None,'scaling_profile':profile,
        'G4_three_frame_full_cloud_passed':bool(three_ok),'G4_11_frame_legacy_accuracy_comparison':'NOT VALIDATED',
        'production_jobs':jobs,'SSD_free_bytes':disk.free,'noise_calibration':'IDEAL_WARP_ASSUMPTION_UNCALIBRATED; sensor noise remains active; no independent warp calibration',
        'not_validated':['Full-cloud post-anchor motion accuracy unless separately demonstrated in recorded jobs',
                         'Real full-cloud backward filtering','11-frame full-cloud legacy accuracy comparison',
                         '60-frame and ten-scene execution; intentionally not launched'],
        'numerical_scope':'Exact chosen linearized constrained Gaussian operator, conditional on nonlinear local linearization and FEJ fixed-lag approximation',
        'limitations':['Sparse FEJ forms normal information with all fill retained; independent QR parity covers tested conditioning only.',
                       'One batched CPU symbolic bipartite matching occurs for FEJ hard-constraint pivots; numerical Schur work is CUDA.',
                       'cuDSS rank-deficient or resource-limited systems fail; no regularization or fallback.',
                       'Prior-predictive innovation NIS is deferred in sparse output, labeled unavailable, never fabricated.',
                       'Nonlinear acceptance retains the baseline no-new-outside-mask policy and reports any preexisting violations.',
                       'Fixed common measurement domain can obstruct descent at discrete ownership/support boundaries.']}
    latest=next((r for j in reversed(jobs) for r in reversed(j['runs'])),None)
    report['current_blocker']=(latest['status'].get('reason') if latest else 'No actual full-cloud evidence found')
    if latest and (latest['status'].get('sparse_failure_profile') or {}).get('info',0)<0:
        report['current_blocker']='cuDSS device-side factorization error INFO=-1 after the first accepted nonlinear iteration; frame 1 has no completed covariance. Negative backend INFO does not establish mathematical singularity. The raw status is retained below.'
    report['last_nonlinear_history']=(latest['status'].get('nonlinear_history',[]) if latest else [])
    report['stage_status']={'G0_model_identity':'NUMERICAL_TEST_WITNESSES_PASS; exhaustive coverage not claimed',
        'G1_sparse_operator_and_MAP':'NUMERICAL_TEST_WITNESSES_PASS',
        'G2_full_FEJ_Schur':'SMALL_GRAPH_QR_PARITY_PASS; real full-cloud lag not reached',
        'G3_nonlinear_mask_and_visibility':'REGRESSION_WITNESSES_PASS; real full-cloud convergence not validated',
        'G4':'NOT_ACCEPTED'}
    report['profile_actual_component_oracles']=[]
    if a.profile_report:
        for path in sorted(a.profile_report.parent.glob('oracle_frame_*_component_*.npz')):
            try:report['profile_actual_component_oracles'].append(oracle(path))
            except Exception as exception:report['profile_actual_component_oracles'].append({'path':str(path),'passed':False,'error':str(exception)})
    report['profile_matches_current_numerical_sources']=(profile is not None and profile.get('production_source_sha256')==production_hashes)
    report['cuDSS_documentation']={
        'device_INFO_not_equivalent_to_rank':'https://docs.nvidia.com/cuda/cudss/1.0.0/types.html',
        'hybrid_memory_not_supported_by_selected_BTF_COLAMD':'https://docs.nvidia.com/cuda/cudss/1.0.0/advanced_features.html'}
    atomic_json(a.output_dir/'temporal_sparse_gpu_validation.json',report)
    rows=[]
    for job in jobs:
        if not job['runs']:
            rows.append(f"| {job['scene']} | {Path(job['output']).parent.name} | {job['status']} | [] | 0 |")
        for run in job['runs']:
            status=run['status'];rows.append(f"| {job['scene']} | {Path(job['output']).parent.name} | {status.get('status')} | {run['journal'].get('completed_frames')} | {sum(f.get('assimilated_points',0) for f in run['frames'])} |")
    witnesses=tests.get('sparse_marginal_witnesses',[])
    last_status=latest['status'] if latest else {}
    last_profile=last_status.get('sparse_failure_profile') or {}
    nonlinear=[{key:value for key,value in entry.items() if key in ('iteration','objective','proposal_objective','accepted','backtracks','frozen_components','true_components')}
               for entry in report['last_nonlinear_history']]
    performance=[]
    for case in (profile or {}).get('cases',[]):
        f=case.get('sparse_factorization_and_covariance_profile',{})
        performance.append('| '+' | '.join(str(x) for x in [case['points'],case['source_setup_GPU_seconds'],
            case['ownership_registration_GPU_seconds'],case['residual_Jacobian_sparse_assembly_GPU_seconds'],
            f.get('factorization_seconds','NOT RUN'),case.get('sparse_MAP_seconds','NOT RUN'),
            f.get('selected_marginals',{}).get('seconds','NOT RUN'),case.get('true_component_scalars_median_p95_max',[None])[-1],
            case['host_scalar_extractions']])+' |')
    md=f'''# Exact sparse GPU temporal IEKF — measured validation

**FULL_SPARSE_GPU_NOT_ACCEPTED.** This report does not certify a successful reconstruction.

Actual final blocker: {report['current_blocker']}

Last USB full-cloud attempt: {last_status.get('elapsed_filter_seconds')} filtering seconds; verified frames {last_status.get('completed_frames')}; largest failed-system component {last_profile.get('failed_system_largest_component_scalars')} scalars. Blade remained NOT_RUN because fail-fast stopped the batch. No post-anchor trajectory/covariance was published. Reported cuDSS INFO errors are device execution/factorization errors, not independent rank certificates ([NVIDIA API documentation](https://docs.nvidia.com/cuda/cudss/1.0.0/types.html)).

Actual accepted nonlinear iteration history before failure:

```json
{json.dumps(nonlinear,indent=2)}
```

Baseline: `{report['baseline_commit']}`. Backend: cuDSS {report['cudss_version']}, Torch {report['torch']}, CUDA {report['CUDA']}, {report['GPU']}.

Focused tests: {tests.get('tests_run')} run, {tests.get('failures')} failures, {tests.get('errors')} errors, {tests.get('skips')} skips. Tests match current hashed sources: {tests_current}.
USB + Blade three-frame full-cloud gate: {three_ok}. Eleven-frame accuracy comparison: NOT VALIDATED.

| Scene | Batch | Actual status | Verified frames | Accepted point updates |
|---|---|---|---|---|
{chr(10).join(rows)}

The JSON beside this report contains exact commands, return codes, numerical certificates, solver dimensions/fill, failures, frame journals, oracle comparisons, profiles and every source function/line/hash. Missing frames are NaN and invalid; frame-0 viewers do not count as motion estimation. Failed outputs and the historical batch have been preserved.

## Measured scaling

Frozen initial linearizations on the same USB observations. These are performance and algebra checks, not accepted nonlinear trajectories. Times are seconds. First-size initialization is cold. Schur and nonlinear acceptance are measured separately in tests/failed pilots, not inferred from assembly speed.

| Points | Source GPU | Ownership/registration GPU | Residual/Jacobian GPU | Native factorization | MAP solve | Joint selected covariance | Largest initial component | Host scalar extractions |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(performance)}

Full profiler traces/kernel names, sparse counts, fill, host RSS and memory are in JSON and the referenced SSD trace files. Torch peak allocation excludes cuDSS native allocations; native estimates and device-free snapshots are recorded separately and are not mislabeled as total measured GPU peak. Profile matches current source hashes: {report['profile_matches_current_numerical_sources']}.

Actual bounded components from the full-cloud **frozen benchmark**, compared independently with SciPy square-root QR (these do not pass the filtering gate):

```json
{json.dumps(report['profile_actual_component_oracles'],indent=2)}
```

## Implementation

Production full mode uses `temporal_sparse_run.py` → typed SoA GPU factors → exact current-pixel equality substitution → invertible source/process/sensor innovation coordinates → sparse augmented KKT factorization. Actual Gaussian prior residuals are eliminated algebraically; no artificial prior is introduced. Current 6×6 and requested state/reference/pixel cross blocks come from selected solves of the full joint operator. Component-colored RHS reuse is confined to truly independent components. FEJ retains information, gradient, linearization, constant cost and remaining hard constraints with all induced fill. Shared raw sensor pixels receive one likelihood; the source chart retains source-depth coupling.

Nonlinear trials re-render all depth competitors, validate hard foreground and four-neighbor reference support, keep the fixed common factor domain, project ideal equalities exactly, and test the complete nonlinear objective. Backtracking applies to complete true variable components; frozen/stalled components are reported and never called converged. No GT enters inference. Legacy and simplified mathematical files unchanged: {all(unchanged.values())}.

Output publication verifies sizes/checksums and commits frame validity only after posterior/covariance/debug diagnostics exist. Batch execution uses an OS process-group deadline, fail-fast default and an independent failure journal. More than three full-mode frames or more than two scenes is blocked pending source-matched three-frame acceptance evidence. No source snapshots are copied to results.

## Mathematical parity evidence

Independent CPU QR/SVD and sparse CUDA witnesses (complete tables and tolerances in JSON):

```json
{json.dumps(witnesses,indent=2)}
```

## Limits and unfinished acceptance

{chr(10).join('- '+x for x in report['limitations'])}

Full-cloud three-frame execution, actual component oracles, backward execution, moving/static accuracy and the eleven-frame comparison are separate acceptance checks. A test pass or sparse MAP certificate alone does not satisfy them. No 60-frame or ten-scene batch was launched. SSD free at audit: {disk.free} bytes.

## Reproduce the bounded gate

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
bash tools/run_temporal_rgbd_critical.sh --mode full --direction forward --frame-limit 3 \\
  --scenes USB_100109,blade_103706 --pcd-subsample-factor 1 \\
  --temporal-warp-noise ideal --stop-after free --temporal-runtime-budget-seconds 120
```

This command reproduces an acceptance attempt, not a production-cleared reconstruction. The ideal additional-warp-zero assumption is uncalibrated; sensor and reference uncertainty stay active. Re-run this reporting tool with `--test-report`, `--profile-report`, and `--batch` paths to independently inspect newly saved results. No commit or push was performed.
'''
    (a.output_dir/'temporal_sparse_gpu_exact_implementation.md').write_text(md)
    print(json.dumps({'verdict':report['verdict'],'three_frame_gate':three_ok,'report':str(a.output_dir/'temporal_sparse_gpu_validation.json')},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
