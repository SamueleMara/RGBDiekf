#!/usr/bin/env python3
"""Write the implementation/acceptance report from actual logs and JSON witnesses.
Never supplies missing experiment values or turns a preflight into reconstruction.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io import json_write

FILES=['train.py','debug/projective_debug.py','debug/assets/projective_common.js',
       *['estimator/'+n+'.py' for n in ['temporal_zbuffer','temporal_reference','temporal_rgbd_terms','temporal_noise','temporal_full_graph','temporal_simplified_iekf','temporal_run']],
       'debug/temporal_debug.py',*['tests/test_temporal_'+n+'.py' for n in ['warp','full_graph','rgbd_pipeline','motion']],
       *['tools/'+n+'.py' for n in ['test_temporal_rgbd','verify_temporal_legacy_parity','audit_temporal_inputs','run_temporal_rgbd_critical','evaluate_temporal_rgbd','compare_temporal_modes','report_temporal_rgbd_validation']],
       'tools/run_temporal_rgbd_critical.sh','configs/temporal_warp_noise.schema.json',
       'tools/f1_source_responsibility_audit.py','tools/f1_joint_responsibility_audit.py','tools/f1_definitive_drift_forensics.py',
       'README.md','docs/PIPELINE_CODE_MAP.md','docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--test-results',type=Path,default=Path('tests/temporal_test_results.json'))
    ap.add_argument('--suite-log',type=Path,required=True);ap.add_argument('--geometry-audit',type=Path,required=True)
    ap.add_argument('--visual-output',help='published geometry-preflight dashboard, if available');a=ap.parse_args()
    tests=json.loads(a.test_results.read_text());geo=json.loads(a.geometry_audit.read_text());datasets=json.loads(Path('reports/temporal_rgbd_baseline/dataset_preflight.json').read_text())
    parity=json.loads(Path('reports/temporal_rgbd_baseline/parity.json').read_text());log=a.suite_log.read_text()
    failnames=re.findall(r'^FAIL: ([^\n]+)',log,re.M);summaries=re.findall(r'Ran (\d+) tests in ([\d.]+)s',log)
    source={}
    for name in FILES:
        p=Path(name);data=p.read_bytes();syms={}
        if p.suffix=='.py':
            tree=ast.parse(data)
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.col_offset==0:syms[node.name]=[node.lineno,node.end_lineno]
        source[name]={'sha256':hashlib.sha256(data).hexdigest(),'lines':len(data.splitlines()),'top_level_symbols':syms}
    machine={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'dirty_status':subprocess.check_output(['git','status','--short'],text=True).splitlines(),
             'sources':source,'new_tests':tests,'legacy_parity':parity,'full_suite':{'summary':summaries[-1] if summaries else None,
             'failures':failnames,'log':str(a.suite_log),'status':'FAIL' if failnames else 'UNKNOWN'},
             'geometry_preflight':geo,'datasets':datasets,'visual_output':a.visual_output,'acceptance':'NOT_SATISFIED',
             'blockers':['MISSING_TEMPORAL_WARP_NOISE in all ten scenes','Existing full-suite failures in archive/legacy paths',
                         'Real temporal estimation, post-update graph sizes, runtime and accuracy not validated']}
    json_write('reports/temporal_rgbd_validation.json',machine)
    m=tests.get('numerical_witnesses') or {};gpu=m.get('CUDA_mean_covariance_discrepancy') or {}
    first=geo['frames'][0];res=first['component_preflight']
    text=f'''# Temporal RGB-D implementation and measured validation

Status: **NOT accepted as a complete full-scene reconstruction implementation.**
The opt-in model and exact feasible-component numerical path exist, with the
limitations below. No real-scene temporal reconstruction completed and no
full-scene production job was launched. Geometry preflights are not reconstruction.

Git commit: `{machine['git_commit']}`; dirty working tree deliberately preserved.
Pre-edit status, dirty patch and CLI help are in `reports/temporal_rgbd_baseline/`.
Exact file hashes, line counts and top-level symbol ranges are in
`reports/temporal_rgbd_validation.json`. New files are separate modules; legacy
remains the CLI default. Earlier untracked temporal IEKS modules were preserved
and are not relabeled as this model.

## Actual executed tests

- Latest focused CPU/CUDA suite: **{tests['tests_run']} tests**, {tests['failures']} failures,
  {tests['errors']} errors, {tests['skips']} skips; {tests['seconds']:.3f} seconds.
- Both modes, both directions, four-frame CPU/CUDA runs include an actual lag
  slide. Separate train.py integration checks both directions and the full
  existing HTML export. These are named tiny fixtures, not real reconstruction.
- Captured pre-edit dirty baseline vs current legacy default: **{len(parity['checks'])}
  arrays and {parity['npz_snapshots_equal']} NPZ archives bit-for-bit equal**, both directions,
  three frames, including full covariance snapshots. Six preexisting tracked
  non-CLI source files are byte-identical to the captured pre-edit tree.
- Required broad discovery command executed: {summaries[-1][0] if summaries else 'unknown'} tests in
  {summaries[-1][1] if summaries else 'unknown'} seconds; **{len(failnames)} failures remain**.
  See the complete log and failure list below. Later numerical changes were
  rechecked by the focused suite; do not interpret this as a green full suite.

| Numerical witness | Actual maximum absolute discrepancy |
|---|---:|
| CPU dense inverse vs sparse component covariance | {m.get('CPU_dense_vs_sparse_component_covariance_max_abs')} |
| JVP/VJP dot-product identity | {m.get('JVP_VJP_dot_product_abs')} |
| Exact lag Schur marginal covariance | {m.get('Schur_marginal_covariance_max_abs')} |
| Two-time analytical vs central finite difference | {m.get('two_time_FD_max_abs')} |
| Tiny CPU/CUDA mean | {gpu.get('mean_max_abs')} |
| Tiny CPU/CUDA covariance | {gpu.get('covariance_max_abs')} |

The full physical 6D tiny-pipeline CPU/CUDA covariance discrepancies, known-motion
results and per-check tolerances/durations are in `tests/temporal_test_results.json`.
Tests include both-state and latent-pixel derivatives, source-depth coupling,
shared-reference cross-point covariance, corrected correlated NIS, log-determinant
and covariance identities under lag elimination, exact no-data publication,
conditioning the reference out, source identity, hard support validity, camera
handover and stale references, deterministic ownership, and positive/negative
motion across reference cells. Monte Carlo NIS/NEES checks use a declared known
Gaussian generative model, not the real-scene noise model.

## Implemented mathematics and preserved behavior

See `docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md` for equations and code map.
The state is 6D world XYZ/velocity; original signed-time F and white-acceleration Q
are reused. Both modes call the same first-hit owner, camera-local lifecycle,
strict four-neighbor foreground sampler and two-time temporal operator.

Full mode has one clean RGB-D latent per unique raw key and one sensor likelihood
per key, regardless of reuse. Source XYZ is generated from that same clean depth,
pixel coordinate error and an independent velocity prior; it is not an additional
independent Pxx observation. The source chart is numerically centered on the
recorded canonical mean. Sparse J/JT operate on CUDA/CPU; exact QR solves and
square-root Schur projection retain all induced cross blocks inside feasible
linearized components. Exactly zero blocks may be omitted; no small nonzero
coupling is thresholded away. Current marginals stream across independent
components; full covariance is cached for configured traces and can be refactored
on demand. FEJ priors preserve marginalized evidence; outside-lag references are
unusable, not deterministic old points. No new process or physical movement gates
were added. Simplified covariance is explicitly approximate and conditional.

Reference interpolation cells refresh after nonlinear proposals. Newly touched
sensor pixels are registered once and old sensor factors are kept, so changing
support cannot erase a bad observation to improve cost. Active current-pixel /
owner changes and invalid reference support remain conservatively rejected.
Hard sort and boundary cases are piecewise approximations, with convergence and
invalidity diagnostics; no claim of global multimodal Bayesian inference is made.
No background RGB-D enters a factor. Existing background feasibility violations
are reported rather than repaired by an invented mask force. In full mode, lack
of a direct row is distinct from an indirect coupled marginal update.

A separate import bug was corrected in three forensic drivers: importing them,
or calling their mocked CLI from unittest, must not disable the caller's CUDA.
Their CPU-only command-line behavior remains. No legacy estimator mathematics
was changed to force its tests to pass.

## Real input and resource evidence

All ten actual synthetic scene folders exist, each with 60 frames, eight cameras,
and 59 Hz from `metadata/timing.json:fps`; no manual timing override is necessary.
No scene contains the required separately justified temporal-warp covariance.
Existing sensor MAD / cross-camera mismatch estimates are different quantities.
No invented positive R_warp was substituted. A real three-frame train preflight
returns `MISSING_TEMPORAL_WARP_NOISE` before optimization, as permitted by spec 16.
An explicit user-assumption file is acceptable; laboratory calibration is not
required. Required format: `configs/temporal_warp_noise.schema.json`.

Storage geometry-only preflight processed all **{geo['points']}** source points and
all eight cameras at frames 0/1/2. CPU/CUDA owner disagreements were
{[f.get('CPU_CUDA_owner_disagreement_pixels') for f in geo['frames']]}, peak allocated CUDA bytes
{max(f.get('cuda_peak_allocated_bytes',0) for f in geo['frames'])}, total wall time
{geo['seconds']:.3f} seconds. Hardware: {tests.get('gpu')}, reported device memory
{gpu.get('device_total_memory_bytes')} bytes.

**Correction of the initial resource estimate:** counting zero-weight neighbors
incorrectly merged the whole nonlinear footprint. That 346,416-point / 276 TB
estimate is withdrawn for the initial linearization. With only exactly nonzero
weights, the preflight finds **{res['components']} components, largest
{res['largest_point_components'][0]} points**. This is unchanged-cloud geometry,
not a post-estimation operator. Components may merge once motion changes weights.
No full inference resource conclusion follows from this preflight. The backend
still uses dense QR within bounded components, generic Python factor orchestration,
and may be slow or hit FULL_RESOURCE_LIMIT after actual updates. Full production
performance and sparse-prior fill have not been validated. Do not call that a
successful result or silently switch to simplified mode.

The Storage source-camera collision is real: point 8444 originates from camera 0,
pixel 151962 = (282,237), z=3.427461862564087 m. Unrestricted rendering there picks
45588, from camera 1/pixel 148711, z=3.4273588334329204 m. Both source identities
remain separate from rendered ownership. The measured three-frame collision
records and images are in the geometry audit; no optimized motion is implied.

## Acceptance status and remaining work

| Contract requirement | Status |
|---|---|
| Legacy default/source preservation and same-seed array parity | PASS on captured dirty-baseline fixture |
| Entire existing suite green | FAIL; list below, no thresholds weakened |
| Shared renderer, foreground validity, camera reference lifecycle | Tested tiny CPU/CUDA; all-point real geometry preflight |
| Full source/noise coupling and true current marginals | PASS on numerical fixtures |
| Actual per-camera full/simplified filtering + lag slide | PASS on tiny fixtures only |
| Actual positive/negative motion and new raw support | PASS on declared toy model only |
| Complete mandatory adversarial/physical calibration coverage | NOT exhaustive / NOT accepted |
| Real 3–11-frame temporal optimization and critical-point drift | NOT RUN: required R_warp missing |
| Real full-scene runtime/covariance/accuracy | NOT RUN / not validated |
| HTML image/3D diagnostics | Implemented; actual geometry HTML rendered in Chrome |
| Source-code snapshots in SSD result folders | Not created |

Full-scene scientific acceptance is therefore NOT satisfied. The missing warp
model and existing-suite failures must be resolved, and real forward/backward
accuracy and component-size evolution measured before promotion. Depth sort is
not physical correspondence; local translation can fail under rotation, occlusion
or silhouettes. Covariances are conditional local Gaussian approximations, not
independently calibrated physical accuracy. The generic factor orchestration is
not a validated high-throughput production CUDA implementation.

## Broad-suite failures (unmodified legacy/archive paths)

'''
    text+='\n'.join('- `'+n+'`' for n in failnames)+'\n\n'
    text+='The archived USB covariance-parity gate remains strict. Other failures concern\nlegacy rigid/joint/support behavior and the legacy rendered-mask static case.\nThose paths were not numerically redesigned by this F1/B1 task.\n\n'
    text+='## Reproduction\n\n```bash\nconda activate rigeuler3dgs\ncd "$HOME/Joint3DGS/RGBDekf"\npython -B tools/test_temporal_rgbd.py\npython -m unittest discover -s tests -t . -v\nbash tools/run_temporal_rgbd_critical.sh --mode full,simplified --dry-run\n```\n\n'
    text+='After supplying a justified warp file, the operator command for ALL ten scenes,\nALL frames/cameras and both directions (not pre-cleared as successful full inference):\n\n```bash\nbash tools/run_temporal_rgbd_critical.sh \\\n  --mode full,simplified --direction both \\\n  --temporal-warp-noise config \\\n  --temporal-warp-noise-config "/absolute/path/to/approved_temporal_warp_noise.json"\n```\n\n'
    text+='The driver defaults to the actual synthetic data root and requested external SSD\nresults parent, creates a unique timestamped child, logs commands/exit codes, and\nnever copies a source snapshot. Add `--frame-limit 11 --direction forward` for the\npilot. Full runs have no frame limit; backward then anchors at original frame 59.\nLive pages: `SCENE_MODE/free_DIRECTION/temporal/index.html`; completed full viewer:\n`SCENE_MODE/debug/index.html`.\n\n'
    if a.visual_output:text+=f'Reviewed geometry-only HTML: `{a.visual_output}`.\n\n'
    text+='Report regeneration:\n\n```bash\npython tools/report_temporal_rgbd_validation.py \\\n  --suite-log /absolute/completed_suite.log \\\n  --geometry-audit /absolute/geometry_preflight/audit.json\n```\n'
    Path('reports/temporal_rgbd_implementation.md').write_text(text)
    lines=['# Critical synthetic scenes: actual status','','No temporal reconstruction was run on these real scenes. Missing values are not imputed.','',
           '| Scene | Frames | Cameras | FPS | Warp covariance | Temporal filtering / accuracy |','|---|---:|---:|---:|---|---|']
    for d in datasets:lines.append(f'| {d["scene"]} | {d["frames"]} | {d["cameras"]} | {d["fps"]} | {"found" if d["warp_noise_found"] else "MISSING"} | NOT RUN |')
    lines+=['','Storage has a complete-point-count geometry/ownership preflight at frames 0/1/2. It is NOT a motion estimate.',
            'No GT enters inference. Static drift, moving-point error, empirical uncertainty coverage and NEES on these real scenes are NOT measured.',
            'Use evaluate_temporal_rgbd.py and compare_temporal_modes.py after actual completed runs; GT evaluation is opt-in and separate.']
    Path('reports/temporal_rgbd_critical_scenes.md').write_text('\n'.join(lines)+'\n')
    print('reports/temporal_rgbd_implementation.md')

if __name__=='__main__':main()
