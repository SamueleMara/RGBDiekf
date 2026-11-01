#!/usr/bin/env python3
"""Validate and report saved fifth-pass evidence. Never runs an estimator/replay."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
from tools.f1_definitive_evidence import MANDATORY,read_json,write_json,csv_read,digest,assert_protected


def number(value):
    if value is None:return 'UNAVAILABLE'
    if isinstance(value,(list,dict)):return json.dumps(value,separators=(',',':'))
    return str(value)


def evidence_reference(filename,index):return f'`{filename}` row {index+2}'


def generate_test_gate_failure(out,tests):
    """A failed test gate is a reportable blocker, never a root-cause verdict."""
    target=Path(out)/'TEST_GATE_FAILURE.md'
    if target.exists():raise FileExistsError(f'Preserving existing report: {target}')
    lines=['# F1 forensic test gate failed','',
           f"Tests run: {tests.get('tests_run')}; failures: {tests.get('failures')}; errors: {tests.get('errors')}; skips: {tests.get('skips')}.",'',
           'Analysis was not started. No accepted causal or reconstruction result is claimed. Existing inference/configuration and earlier outputs are unchanged.',
           '', 'The complete records are in `tests.json` and `tests.log`. Historical-array absence must be distinguished from an actual failed production comparison. Parity tolerances are not relaxed to allow the run to continue.','']
    for r in tests.get('records',[]):
        if r['status']!='PASS':lines.extend([f"## {r['status']}: {r['test']}",'','```text',r.get('detail',r.get('reason','No detail recorded')),'```',''])
    target.write_text('\n'.join(lines)+'\n');return target


def generate(out):
    out=Path(out);target=out/'DEFINITIVE_F1_STATIC_DRIFT_ROOT_CAUSE.md'
    if target.exists():raise FileExistsError(f'Preserving existing report: {target}')
    inventory=read_json(out/'evidence_inventory.json');analysis=read_json(out/'analysis.json');tests=read_json(out/'tests.json');assert_protected(inventory)
    required=['baseline_parity_detailed.csv','point_catalogue_and_selection.csv','camera_row_lifecycle.csv','frame0_vs_onset_raw_images.csv','inner_information_exact.npz','inner_information_index.csv','causal_interventions.csv','depth_surface_evidence.csv','rgb_appearance_evidence.csv','observability_identifiability.csv','numerical_branch_stability.csv','static_full_timeline.csv','negative_controls_moving.csv','tests.log']
    missing=[f for f in required if not (out/f).exists()];lifecycle=csv_read(out/'camera_row_lifecycle.csv');parity=csv_read(out/'baseline_parity_detailed.csv');interventions=csv_read(out/'causal_interventions.csv');forces=csv_read(out/'inner_information_index.csv');raw=csv_read(out/'frame0_vs_onset_raw_images.csv');obs=csv_read(out/'observability_identifiability.csv');times=csv_read(out/'static_full_timeline.csv')
    dossier={(r['scene'],int(r['point_id'])):r for r in analysis['dossiers']}
    uncovered=[f'{scene}/{pid}' for scene,ids in MANDATORY.items() for pid in ids if (scene,pid) not in dossier]
    unavailable=[f'{s}/{p}' for s,ids in MANDATORY.items() for p in ids if dossier.get((s,p),{}).get('parity_status') in ('UNAVAILABLE',None)]
    missing_input=[name for name,record in inventory['required'].items() if record['status']!='AVAILABLE']
    row_errors=[r for r in lifecycle if r.get('used_in_solver')=='True' and r.get('reason_used') not in ('VALID','')]
    closure_errors=[r for r in forces if r.get('attribution_closure') and float(r['attribution_closure'])>1e-9]
    # Known non-parity cases are qualified, not hidden. Missing decisive arithmetic
    # prevents a definitive within-model result for either required primary point.
    primary_ok=all(dossier.get((s,p),{}).get('parity_status')=='PARITY_PASS' for s,p in [('storage_45135',8444),('USB_100109',67827)])
    accepted=bool(tests.get('successful')) and not missing and not uncovered and not unavailable and not missing_input and not row_errors and not closure_errors and primary_ok
    for scene,pid in [('storage_45135',8444),('USB_100109',67827)]:
        inner=[r for r in forces if r['scene']==scene and int(r['point_id'])==pid and r.get('iteration')!='accepted']
        if not inner or any(r['evidence_level']!='EXACT_CODE_AND_ARITHMETIC' or float(r['delta_position_error_m'])>2e-12 or float(r['delta_velocity_error_m_s'])>1e-10 for r in inner):accepted=False
        d=dossier.get((scene,pid),{})
        if np.linalg.norm(np.asarray(d.get('velocity_transfer_error_m_s',[np.inf])))>1e-9:accepted=False
    primary_controls=[(s,p) for s,p in [('storage_45135',8444),('USB_100109',67827)] if not any(r['scene']==s and r['point_id']==str(p) and r.get('arm')=='T1_ALL_IMAGES_FRAME0' for r in interventions)]
    if primary_controls or not analysis['protected_inference_unchanged']:accepted=False
    verdict={'accepted':accepted,'meaning':'Accepted forensic evidence package conditional on stated identifiability limits; never a production-fix endorsement','tests_successful':tests.get('successful'),
             'missing_outputs':missing,'unavailable_mandatory':unavailable,'unavailable_prior_evidence':missing_input,'uncovered_mandatory_points':uncovered,'invalid_camera_force_rows':len(row_errors),'additive_closure_errors':len(closure_errors),'primary_parity_pass':primary_ok,'missing_primary_frozen_image_controls':primary_controls,
             'inference_unchanged':analysis['protected_inference_unchanged'],'known_qualified_cases':[f"{r['scene']}/{r['point_id']}" for r in analysis['dossiers'] if r['parity_status']!='PARITY_PASS'],
             'physical_unique_bias_identified':False,'limitations':analysis['limitations']}
    write_json(out/'report_validation.json',verdict)
    lines=['# Definitive F1 static-drift causal forensics','',f"Evidence-package status: **{'VALIDATED_WITH_QUALIFICATIONS' if accepted else 'INCOMPLETE_OR_INDETERMINATE'}**. Tests: {tests['tests_run']} run, {tests['failures']} failures, {tests['errors']} errors, {tests['skips']} skips.",
           '', 'This report concerns the archived standard directional forward F1 `[x,v]` IEKF only. No inference source/default/noise was changed; no full scene was re-filtered. All counterfactuals are copied CPU point replays. GT is selection/evaluation only. A physical bias, face identity or new estimator is not established by algebraic stabilization.',
           '', '## Traceable evidence table','', '| Scene / point | Parity | First accepted correction: position / velocity | Accepted cameras | Evidence |','|---|---|---|---|---|']
    for key,d in sorted(dossier.items()):
        match=next(((i,r) for i,r in enumerate(forces) if r['scene']==key[0] and int(r['point_id'])==key[1]),None)
        ref=evidence_reference('inner_information_index.csv',match[0]) if match else 'INDETERMINATE: no arithmetic record'
        lines.append(f"| {key[0]} / {key[1]} | {d['parity_status']} | {d.get('dx_norm_m','UNAVAILABLE')} m / {d.get('dv_norm_m_s','UNAVAILABLE')} m/s | {d.get('accepted_rows','UNAVAILABLE')} | {ref}; {d.get('verdict_status')} |")
    lines+=['','## Input and repository audit','',f"Commit: `{inventory['git_commit']}`; branch: `{inventory['git_branch']}`. Existing dirty changes were preserved and hashed. Exact diff and source protection hashes: `evidence_inventory.json`.",
            'The seed for each random holdout is derived from its baseline manifest SHA256 before residual inspection. Holdout selections and population denominators are separate from hand-picked failure cases. Untraced holdouts are explicitly unavailable for parity-qualified causal claims.',
            '', '## Errata and contradictions','',
            '- The original reports are unchanged. Cross-reference second-pass `SECOND_PASS_ROOT_CAUSE.md`, third-pass `THIRD_PASS_CAUSAL_RESULT.md`, fourth-pass `FOURTH_PASS_CAUSAL_REPORT.md` and `diagnostic_units_corrections.csv` via inventory hashes.',
            '- USB 67827 camera 4 at frame 1, Storage 262894 cameras 3/4 and Blade 14104 camera 0 cannot be named as update causes when frozen/excluded. Only the saved production-admitted rows enter information attribution. USB 67827 can have both cameras 5 and 6 admitted; camera 6 being the offending depth row does not imply it was the sole row.',
            '- Historical mixed-six-state numbers have no position unit. Position (m), velocity (m/s), and covariance blocks are compared separately. A3 zero residual at unchanged xref is an identity, not a fitted physical noise/bias model.',
            '- Runner correction: historical arrays can contain only five frames; later frames are marked NO_ARCHIVED_FRAME and still checked against production. Additional per-block Frobenius diagnostics (1e-12 absolute in block units plus 1e-9 relative) do not override the original strict entrywise covariance gate. All original parity tolerances are unchanged; no bitwise covariance equality is asserted.',
            '- A4 per-frame R+R0 is a sensitivity probe. Its missing cross-time covariance differs from the shared-reference GLS/latent nuisance oracle. Source backprojection/color error correlations remain unknown.']
    for key,d in sorted(dossier.items()):
        scene,pid=key;select=lambda rows:[(i,r) for i,r in enumerate(rows) if r.get('scene')==scene and r.get('point_id')==str(pid)]
        pp=select(parity);rr=select(raw);ii=select(interventions);oo=select(obs);tt=select(times)
        lines+=['',f'## Dossier: {scene} / {pid}', '',f"**{d['parity_status']} / {d.get('role')} / {d.get('verdict_status')}**",'','### Source and first update','',
                f"GT grounding / selection (evaluation only): {number(d.get('selection_and_GT_evaluation_only'))}.",
                f"Source frame/camera/pixel: {number(d.get('source'))}; xref: {number(d.get('xref'))}; canonical encoded RGB: {number(d.get('canonical_color'))}. Source covariance and ID mapping: `{scene}_{pid}_references.json` and inventory point mapping. GT grounding is nearest labelled cloud evidence, not exact material-point identity.",
                f"First accepted nonzero correction at frame {d.get('first_update_frame','UNAVAILABLE')}: Δx={number(d.get('dx_m'))} m, ||Δx||={d.get('dx_norm_m')} m; Δv={number(d.get('dv_m_s'))} m/s, ||Δv||={d.get('dv_norm_m_s')} m/s. Accepted cameras: {d.get('accepted_rows')}.",
                f"Acceptance: {d.get('acceptance')}; converged: {d.get('converged')}; J={d.get('accepted_cost')} versus its actual threshold {d.get('accepted_cost_threshold')}. Fixed accepted-camera objective before/after: {number(d.get('accepted_original_common_cost_before'))} / {number(d.get('accepted_original_common_cost_after'))}.",
                '', '### Exact arithmetic and nonlinear path','',
                f"Velocity transfer Pvx Pxx⁻¹={number(d.get('velocity_transfer_matrix_per_s'))} s⁻¹; residual Δv−Pvx Pxx⁻¹Δx={number(d.get('velocity_transfer_error_m_s'))} m/s. Full P−, P+, six-state iterate, H, R, S, ν, A, b, eigenvectors, force components and prior term are indexed in `inner_information_exact.npz` / `inner_information_index.csv`.",
                f"Accepted-state next stationary step: {number(d.get('accepted_next_stationarity_step'))}. This is a next local step, not the accumulated accepted displacement. A nonconverged accepted iterate need not be a minimum. Its cost-adequacy test is distinct from monotone descent.",
                'Full raw/validity/frozen-owner/candidate/admitted/accepted lifecycle: `camera_row_lifecycle.csv`; exact prior and measured objective/cell changes: `iteration_objectives.csv`. Raw six-state eigenvectors are not interpreted as physical directions; prior-whitened dimensionless and position-only information eigensystems are saved separately.']
        if pp:lines.append('Baseline parity: '+', '.join(evidence_reference('baseline_parity_detailed.csv',i) for i,r in pp[:2])+f"; first divergence: {d.get('first_divergent_frame')}.")
        lines+=['','### Temporal evidence and geometric/appearance identity','']
        if rr:
            k=d.get('first_update_frame');causal=[(i,r) for i,r in rr if int(r['frame'])==k and int(r['camera']) in d.get('accepted_rows',[])]
            for i,r in causal:lines.append(f"Camera {r['camera']}: {evidence_reference('frame0_vs_onset_raw_images.csv',i)}; raw RGB cell equal={r.get('raw_2x2_rgb_equal')}, raw depth cell equal={r.get('raw_2x2_depth_equal')}; reference={r.get('reference_status')}; temporal Δr={r.get('delta_r')}. Original and float32-production 2×2 values, weights, gradients, K/T, masks and pixel-branch hex values are in that row.")
        lines+=['`depth_surface_evidence.csv` contains 2×2 arithmetic/inverse sampling and noise-weighted 3×3/5×5 plane fits. A conditional planar fit does not establish face/material identity; unsupported normals are NORMAL_NOT_IDENTIFIABLE. `rgb_appearance_evidence.csv` compares global gain/offset held-out points and local offsets, without converting a conditional score into physical calibration.','', '### Controlled interventions and falsification','']
        if d['parity_status']!='PARITY_PASS' or d.get('role')=='boundary':lines.append('**NUMERICAL_CAUSAL_INTERVENTION_INDETERMINATE**: benefit claims relative to production are blocked. One-sided branch probes remain diagnostic counterexamples only.')
        else:
            at=[(i,r) for i,r in ii if r.get('frame')==str(d.get('first_update_frame'))]
            for i,r in at:lines.append(f"- {r.get('arm')}: Δposition versus A0={r.get('position_delta_vs_A0_m')} m; Δvelocity={r.get('velocity_delta_vs_A0_m_s')} m/s; support={r.get('support_status',r.get('qualification'))}; {r.get('evidence_level')}. {evidence_reference('causal_interventions.csv',i)}.")
        lines+=['','### Persistence, observability and identifiability','']
        if tt:
            last=tt[-1][1];lines.append(f"Read baseline through frame {last['frame']}: final drift={last['drift_norm_m']} m; integrated-velocity vector={last['integrated_velocity_m']} m; correction vector={last['cumulative_correction_m']} m; vector closure={last['closure_vector_m']} m. First all-camera nonmeasurement={last['first_all_camera_nonmeasurement']}; later readmission={last['first_readmission']}.")
        for i,r in oo:lines.append(f"- {r['model']}: {r.get('status')}; rank={r.get('rank')}/{r.get('columns')}; holdout={r.get('heldout_status')}. {evidence_reference('observability_identifiability.csv',i)}.")
        lines+=[f"Physical initiating mechanism: **{d.get('physical_mechanism','INDETERMINATE')}**. Exact within-model residual/gradient forces can be established without uniquely identifying BRDF, calibration, pixel footprint or a hidden face. A proposed bias state is unimplemented and not authorized."]
        image=out/'figures'/f'{scene}_{pid}_dossier.png'
        if image.exists():lines+=['',f'![Measured pixels, update costs and trajectory](figures/{image.name})']
        if (out/'figures'/f'{scene}_{pid}_objective.png').exists():lines+=['',f'![Original fixed-support likelihood](figures/{scene}_{pid}_objective.png)']
        if scene=='USB_100109' and pid==67827 and (out/'figures/USB_100109_67827_camera4_late.png').exists():lines+=['','![Late camera-4 raw observations](figures/USB_100109_67827_camera4_late.png)']
    lines+=['','## Three separate conclusions','',
            '1. **Within-case mechanism:** the point dossiers identify the exact saved admitted rows, force/Jacobian path, accepted displacement and velocity transfer. Only parity-qualified, comparable-support counterfactuals support local causal statements; accepted nonstationary states and support switches are reported explicitly.',
            '2. **Across-case scope:** `population_summary.csv` and the preregistered holdout selections give denominators, camera imbalance, unmeasured points, stable measured controls and reporting drift bands. A failure-selected case series does not estimate a causal prevalence. Untraced holdouts cannot establish general causal rates.',
            '3. **Physical identifiability boundary:** a position perturbation and a camera/channel nuisance can share the same likelihood direction; one late observation cannot independently establish a missing frame-0 offset. The saved actual Jacobian nullspaces identify those ambiguities. Trusted dense material/face correspondences and independently characterized source/cross-time noise would be needed to distinguish several physical explanations.',
            '', '## Proposed alternatives — unimplemented hypotheses only','',
            'A local surface-consistent interpolation, an appearance nuisance, a branch-stable optimizer or an alternative prior would each require the prescribed held-fixed falsification tests. None is an inference change in this work; no alternative is endorsed from a tautological reference subtraction.',
            '', 'See `OPEN_QUESTIONS_AND_DISCONFIRMING_EVIDENCE.md`, `limitations.md`, `tests.log` and `report_validation.json`.']
    if (out/'figures/storage_45135_8416_branches.png').exists():lines+=['','![Separate position and velocity branch sensitivity](figures/storage_45135_8416_branches.png)']
    target.write_text('\n'.join(lines)+'\n')
    contradictions=['# Open questions and disconfirming evidence','', 'These questions prevent a unique physical-cause claim even when within-model arithmetic agrees.','',
        '1. Storage 8416/8417/8418: a projection-cell branch can defeat production parity. No stabilization benefit is admitted without branch-specific reproduction.',
        '2. USB 67827: reference validity and NIS-dependent support can cause late camera-4 readmission. Depth reference is not universal suppression; inspect A2/A3/A4 per-frame gate evidence.',
        '3. USB 87290/85696 moving controls: preserve genuine motion evidence and distinguish rejection/ownership from a physical lack of view. GT columns are evaluation-only.',
        '4. A3 cancels static residuals algebraically. This does not identify a physical bias or prove a calibrated generative model.',
        '5. Local point×camera offsets can absorb geometry. Global camera gain/offset must survive held-out points; a successful in-sample fit alone cannot establish camera calibration.',
        '6. Conditional plane fits do not certify a material face at a crease; invalid normals and source/reference correlations remain explicit.',
        '7. A4 ignores shared-reference cross-time noise. Its trajectory cannot be interpreted as the exact latent-bias posterior.',
        '8. Random holdouts with no archived full covariance/ownership trace remain unavailable for production-validated interventions. Aggregate statistics do not repair this missing evidence.',
        '', '## Invalid or unavailable comparisons','']
    for r in analysis['dossiers']:
        if r['parity_status']!='PARITY_PASS':contradictions.append(f"- {r['scene']}/{r['point_id']}: {r['parity_status']}; {r.get('reason','first divergence '+str(r.get('first_divergent_frame')))}.")
    contradictions+=['- '+r for r in analysis['limitations']]
    (out/'OPEN_QUESTIONS_AND_DISCONFIRMING_EVIDENCE.md').write_text('\n'.join(contradictions)+'\n')
    files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='output_hashes.json'};write_json(out/'output_hashes.json',files)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();path=generate(a.out);print(path);return 0 if read_json(a.out/'report_validation.json')['accepted'] else 3


if __name__=='__main__':raise SystemExit(main())
