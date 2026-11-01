"""Small final-pair report/figures. No production updates or GT imports."""
from pathlib import Path
import numpy as np
from tools.f1_definitive_evidence import read_json


def figures(sc,env,problems,original,U,outcomes,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for c in (0,1):
        cam=sc.cams[c];a=axes[c,0];b=axes[c,1];points=[cam.project(sc.x0[pid])[:2] for pid in env.pids]
        center=np.mean(points,axis=0);x,y=np.floor(center).astype(int);x0,y0=max(0,x-7),max(0,y-7);x1,y1=min(cam.W,x+9),min(cam.H,y+9);im=sc.images(1)[c]
        a.imshow(im.rgb[y0:y1,x0:x1]);b.imshow(np.where(im.mask[y0:y1,x0:x1],im.depth[y0:y1,x0:x1],np.nan))
        for j,(u,v) in enumerate(points):
            for ax in (a,b):ax.plot(u-x0,v-y0,('r+' if j==0 else 'bx'),label=f'{env.pids[j]} predicted')
            pu,pv=int(np.rint(u)),int(np.rint(v));occ=env.occupants(c,pu,pv)
            if occ:a.annotate(f'owner {occ[0]["point_id"]}',(u-x0,v-y0),xytext=(u-x0+1,v-y0+2+j))
        a.set_title(f'camera {c}: actual RGB and rounded ownership');b.set_title('actual masked optical depth (m)');a.legend(fontsize=7)
    fig.suptitle('storage 8444 (source c0/282,237), 45588 (source c1/231,232), frame 1; canonical prior positions, raw J0')
    fig.tight_layout();fig.savefig(out/'figures/storage_8444_45588_camera0_camera1_ownership.png',dpi=130);plt.close(fig)
    from tools.f1_joint_responsibility_audit import PairProblem
    fig,axes=plt.subplots(1,2,figsize=(12,4));grids={}
    for ax,arm,transfer in zip(axes,('J0_SHARED_NOISE_CONTROL','J1_ATOMIC_LEGACY'),(False,True)):
        if arm+'_8444' not in outcomes:continue
        model=PairProblem(sc,env,problems,U,transfer=transfer);da=np.r_[outcomes[arm+'_8444']['dx_m'],outcomes[arm+'_8444']['dv_m_s']]
        db=np.r_[outcomes[arm+'_45588']['dx_m'],outcomes[arm+'_45588']['dv_m_s']];t=np.linspace(-.25,1.25,21);Z=np.full((21,21),np.nan)
        for i,a in enumerate(t):
            for j,b in enumerate(t):
                try:Z[j,i]=model.evaluate(model.sm+np.r_[a*da,b*db])['criterion']
                except (ValueError,np.linalg.LinAlgError):pass
        im=ax.imshow(np.ma.masked_invalid(Z),origin='lower',extent=[t[0],t[-1],t[0],t[-1]],aspect='auto');fig.colorbar(im,ax=ax)
        ax.set(xlabel='fraction of 8444 conditional correction',ylabel='fraction of 45588 conditional correction',title=arm+'\nundefined support omitted')
        grids[arm]=Z
    fig.suptitle('Frame-1 conditional projective criterion within each arm; NOT common fixed-pixel evidence or an assignment ranking')
    fig.tight_layout();fig.savefig(out/'figures/storage_8444_45588_cost_landscape.png',dpi=130);plt.close(fig)
    if grids:np.savez_compressed(out/'figures/conditional_cost_grids.npz',**grids)


def generate(out):
    out=Path(out);target=out/'FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md'
    if target.exists():raise FileExistsError('Preserving '+str(target))
    r=read_json(out/'results.json');archive=read_json(out/'archive_completeness.json');tests=read_json(out/'tests.json');prov=read_json(out/'provenance.json')
    rows=r['outcomes'];within='CONFIRMED_CAUSAL within the faithful J0 numerical model' if r['baseline_gate'] else 'NUMERICALLY_INDETERMINATE: Gate A0/test gate failed'
    lines=['# Final joint pixel-responsibility test: 8444 / 45588','',
        f"Frame-1 8444 baseline gate: **{r['baseline_gate']}**. Tests: {tests['tests_run']} run; failures {tests['failures']}, errors {tests['errors']}, skips {tests['skips']}.",
        '', '**Final fixed-pixel material-association verdict: UNIDENTIFIABLE.** A completed conditional pair response is not a calibrated pixel-identity posterior or a full-sequence recovery.',
        '',f"Commit `{prov['git_commit']}`; dirty state and input/source SHA256 are saved. No production inference/configuration, previous report or whole-scene trajectory was changed.",
        '', '## Archive and reconstruction tier','',
        '45588 is not presumed traced. `archive_completeness.json` lists availability per point/frame. Both frame-1 priors are reconstructed from measured source geometry and the original Pvv/Q/dt. 8444 is independently checked against its saved full trace; 45588 is PARITY_NOT_VERIFIABLE when only float64 XYZ, float32 velocity/variance diagonals and no full 6×6 posterior exist.',
        'Canonical external positions are exactly predicted at frame 1 (initial velocities are zero). External inner-iteration states are not archived globally; the local pair uses the declared frozen environment, TIER_CONTROLLED_LOCAL. Current pair separation after production cannot be inferred from canonical separation.',
        '',f"Canonical separation: {r['canonical_separation_m']*1000:.12g} mm. Camera-0 prior optical-depth gap: {r['source_collision_depth_gap_m']*1000:.12g} mm. These are different quantities, neither proving material identity.",
        '', '## Quantitative frame-1 responses','',
        '| Arm | Point | ||Δx|| mm | ||Δv|| m/s | Camera rows | Status |','|---|---:|---:|---:|---|---|']
    for key,v in rows.items():lines.append(f"| {v['arm']} | {v['point_id']} | {v['dx_mm']:.12g} | {v['dv_norm_m_s']:.12g} | {v['rows']} | {v.get('status',v.get('acceptance'))} |")
    lines+=['','J0 retains the original per-point solver/chi-square fallback rule, including nonconvergence. New joint legacy diagnostics update both states, use shared raw-pixel covariance, one rounded-pixel owner, a fixed prior and strictly decreasing comparable row-support energy. They never silently zero invalid rows. Their covariance is a frozen-covariance GN approximation; algebraically referenced covariance is not a calibrated posterior.',
        'The current-ownership shared-noise control separates the change in noise treatment from atomic ownership. Per-channel declared sensor variances are unchanged; covariance between overlapping interpolants is L diag(var) Lᵀ. Exact repeated observations with singular covariance are reported undefined, not repaired with jitter.',
        f"Solver scope: {r.get('joint_solver_scope','conditional projective diagnostics only')}",
        '', '## Answers to the seven required questions','',
        '1. **Is 45588 the material correspondent? UNIDENTIFIABLE.** The source and complete candidate tables establish its prior depth order and source pixel. A foreground/depth-nearby sample does not identify which material patch generated camera-0 RGB. The fixed-pixel partial RGB/Z scores include Gaussian normalizers, but omit a point-to-pixel location/coverage likelihood. A zero-size point supplies none. No source bonus or OTHER penalty was invented.',
        '2. **Does transferring ownership move error into 45588?** The table reports both points under J0, atomic transfer, bias-only and combined algebraic controls. `ownership_changes.csv` names both owners. A 45588 result with no accepted camera is UNOBSERVED_COAST, regardless of its displacement. Where the new optimizer cannot find a descending physically valid step, the result is numerical indeterminacy, not rescue.',
        '3. **Which assignment has higher common-pixel evidence? UNIDENTIFIABLE.** `joint_assignment_enumeration.csv` enumerates actual local candidates on the same raw pixel IDs, including external points. Every complete model is explicitly UNDEFINED_MODEL: multiplying either finite RGB/Z compatibility by an unconstrained positive missing coverage factor can reverse its odds. Thus complete log-odds can span (−∞,+∞). Partial compatibility must not be used as an assignment posterior.',
        '4. **Reference versus ownership?** J4_CURRENT_ALGEBRAIC retains current eligibility while subtracting a valid point×camera frame-0 residual. J1 changes the single owner atomically and updates both points. Compare their numerical response in the table; unchanged-image reference cancellation is an algebraic identity, not proof of a calibrated camera discrepancy or correct association.',
        '5. **Additive or interacting?** The explicit 12-state ownership×reference response contrast is saved in model_comparison_identifiability.csv, separately for positions and velocities. It is conditional on support and convergence, not a globally additive Bayesian attribution. Exact J0 camera/channel force closure and Pvx Pxx⁻¹ Δx are recorded separately. An uncalibrated missing likelihood cannot yield a unique physical causal split.',
        '6. **Both points validly tracked through frame 10? NOT ESTABLISHED.** No coupled sequence CSV or trajectory plot is emitted. The first missing full 45588 posterior is frame 1; the fixed-observation likelihood and persistent reference/source covariance are also unidentified. `joint_sequence_status.json` gives the exact path/array blocker. A local hard-assignment linearization is not substituted for that missing model.',
        '7. **Minimal production change? Not authorized or established.** This pass changes no production code. A valid design needs an explicit point/material correspondence or finite support observation model, normalized assignment/OTHER likelihood, supported assignment priors and source/reference correlations. The archived uniform source-pixel response is source-ray uncertainty, not a calibrated non-source feature correspondence.',
        '', '## Reference uncertainty and covariance','',
        'LATENT_REFERENCE is NOT_IDENTIFIED for real data. The tests verify proper small joint Gaussian nuisance marginalization against Schur/GLS and a normalized toy mixture with genuine mixed Hessian terms. Those synthetic checks do not calibrate the real nuisance prior. reference_bias_uncertainty.csv gives conditional per-channel Cauchy–Schwarz variance bounds; it does not assemble unknown correlations into a fictitious joint covariance.',
        'Conditional hard assignments can give zero cross-point blocks when observations and priors are disjoint. Shared bilinear noise can give nonzero blocks. Marginalizing assignment uncertainty adds between-component covariance; tests verify that these cross terms survive 12×12 time propagation. No block reset is used.',
        '', '## Falsification and mechanism verdicts','',
        '- Exclusive ownership as the sole cause: **SUPPORTED_BUT_NOT_ISOLATED / not established as sole cause**. Its atomic effect is quantified with both states and observation counts; source-only no-row stability is rejected as success.',
        f'- Camera-1 persistent RGB factor and velocity transfer: **{within}**. The mechanism is the actual admitted gradient/R/prior solve; its physical origin (bias versus wrong material) remains UNIDENTIFIABLE.',
        '- Wrong material match versus valid view-dependent discrepancy: **UNIDENTIFIABLE** from partial fixed-pixel scores and unspecified coverage/nuisance priors.',
        '- Optimizer instability: inspect every accepted step/support failure and convergence status. A failed or constrained local diagnostic remains **NUMERICALLY_INDETERMINATE**.',
        '- heldout_controls.csv records 60019, USB 67827 late camera-4/missing-reference admission, 87290 support collapse, 85696 multiple-competitor suppression and 8416 unit-safe branch probes. None receives a full-scene/factorial optimization or GT recentering.',
        '', '## Evidence','',
        'See baseline_joint_parity.csv, source_collision_candidates.csv, fixed_observation_pixels.csv, candidate_likelihoods.csv, per_point_per_camera_factors.csv, joint_iteration_history.csv, joint_covariance_12x12.npz and matrix_units.json. No cost across different sampled row sets is treated as common fixed-pixel evidence.',
        '', '![Actual camera-0/1 observations and source/owner projections](figures/storage_8444_45588_camera0_camera1_ownership.png)',
        '', '![Conditional pair cost regions; not assignment evidence](figures/storage_8444_45588_cost_landscape.png)']
    target.write_text('\n'.join(lines)+'\n');return target
