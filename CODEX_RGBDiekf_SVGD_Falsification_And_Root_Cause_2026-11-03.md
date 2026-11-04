# CODEX — RGBDiekf SVGD Phase-1: decisive falsification and causal root-cause audit

**Date:** 2026-11-03  
**Repository:** `SamueleMara/RGBDiekf`  
**Audited committed base:** `b62c7ceeb300c852a58f34dd748d5a1ce26d987a`  
**Status:** **INVESTIGATE AND REPORT. Do not implement a new inference model, approve any deviation, or promote experimental inference to production.**  
**Authoritative materials:** `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`, `docs/SVGD_THEORY_TO_CODE_CONTRACT.md`, `docs/SVGD_THEORY_AMENDMENT_PROPOSAL_2026-11-03.md`, `reports/svgd/theory_deviation_register.{json,md}`, and the existing Blade audit results.

## 0. The user-level research question

**Is the current normalized XYZ–RGB feature likelihood capable of preferring the physically correct moving Blade sequence over the stationary sequence, at the dataset's *measured* uncertainty?** If yes, is the failure caused by the motion prior, the proposal/line search, or the geometry of the likelihood basin? If no, the observation model must be reconsidered before any further optimizer or posterior work.

The goal of Phase 1 is **high-confidence, spatially distributed moving seeds**, not perfect trajectories for all points. Full rigid poses, joints, and ambiguous point completion happen later. Absolutely no GT part membership, rigid transformations, joint data, or true trajectories may enter the inference path. They are permitted only inside an isolated, clearly named **post-hoc oracle**.

### Measured findings the investigation must explain (not silently reinterpret)

- Blade `blade_103706`, eight RGB-D cameras, `N=34,369`, GT-moving `3,218` (GT only in post-hoc audit), prismatic displacement `14.16 mm/frame`.
- Two-frame runs: 21/21 attempted Stein backtracks rejected in both `positions_only` and `conditional_temporal_lift` variants. Ten-frame runs: same outcome.
- Ten-frame first step: `ΔE_obs=+1.831e3`, `ΔE_time=+2.827e9`; initial anchor `2.817e9` of that penalty, all nine temporal process intervals together `1.038e7` (~99.6% anchor).
- Smallest 10-frame tested step `eta=2.882e-10 m²`: `ΔE_obs=-6.502e-4`, `ΔE_time=+2.571e-3`. Directional derivative `⟨∇E_obs,P⟩=-2.256e6`, `⟨∇E_time,P⟩=0` at stationary initial state. No assertion of a sign error without checking actual finite differences.
- Canonical source covariance has median eigen-direction standard deviations `[0.0000688, 1.968, 1.968] mm`. Depth sigma `~68.8 nm` matches the uniform float32 depth quantum `2.384e-7 m / sqrt(12)`. **Do not inflate or floor this covariance to obtain motion.**
- Stein field on GT-moving points aligns with physical motion initially (post-hoc median cosine `0.999` at frame 1), but declines by late frames (`0.325` at frame 9; negative around frames 5–6). Target-density spacing `~2.6 mm`; current global Stein/feature bandwidth `sqrt(H_X)~118 mm`.
- `cos(Phi,-grad E_obs)` ≈ `0.032` stacked over 10 frames; per-frame values all ≤ `0.062`. Moving receivers get ~21.8% kernel mass from static sources; ~15.6–17.3% static particles have a Stein field dominated in magnitude by moving-source contributions.
- Prior smaller synthetic tests recovered only ~52% of GT translation displacement and ~8% of hinge motion by the *reported displacement metric*, with measurable false static drift. These statistics are NOT moving-seed precision/recall.
- Existing numerical algebra tests passed. **Algebraic parity, UI QA, and low GPU allocation do not constitute motion-recovery validation.**

## 1. Strict scope and change control

1. Before touching files: `git rev-parse HEAD`, `git status --short`, diff and hash every relevant SVGD source/config. Record the current commit and preexisting modifications. Protect all uncommitted work. The previous parity report listed deletions of older Markdown specifications; do not stage, restore, or delete anything without first reporting the actual state.
2. **Do not edit** `train.py`, working IEKF/UKF, existing rigid/joint stages, main experimental/strict SVGD configs, current `svgd_support_run.py` inference math, or the original theoretical PDF. Do not approve `DEV-0001..0011` or add production DEV-0012/0013. A diagnostic can construct **counterfactual proposals offline** but must not silently change the inference routine.
3. Add narrowly scoped diagnostic files under `tools/` and `tests/`, with reports saved under a new run-specific `posthoc/` directory **without overwriting** prior artifacts. If some operator must be refactored to expose its existing objective, make an exact read-only wrapper and show byte-for-byte / oracle numerical parity.
4. Use the **complete masked clouds** from all cameras. The 30k/60k visual cap must NEVER limit inference. Use a small diagnostic subset only where clearly marked as an independent numerical oracle; report its effect on interpretation. No normals, meshes, surfels, GT correspondences or hard pixel-to-point assignment in the algorithm.
5. Preserve **one unified uncertainty-aware model** for synthetic and real datasets. Dataset-specific sensor/representation/feature/process covariance parameters may differ, with unit-checked provenance. Do not turn off uncertainty for synthetic. Do not tune noise or bandwidths on GT motion/labels. No arbitrary gates, clipping, covariance inflation, or speed thresholds.
6. Report every departure from the frozen theory or production inference as an **offline counterfactual / proposal only**, referencing original equations and their changed mathematical mapping. Do not call an experimental score a calibrated probability.

## 2. Reuse actual code and run files

Review before implementation:

- `estimator/svgd_support_run.py`: `_stages`, `SequenceObjective`, frames, target score, feature objective, line search, objective timeline.
- `estimator/svgd_temporal.py`: `TemporalPrior.energy`, `.grad`, `.velocity_lift`, `.solve`, normalizer; motion process and frame-0 anchor.
- `estimator/svgd_target_score.py`: normalized Gaussian mixture and responsibilities; `estimator/svgd_particle_operator.py`: global all-pairs field.
- `estimator/svgd_feature_measurement.py`: finite RFF map and `FrameFeatureTerm`, frozen `Gamma_t`, exact feature-NLL.
- `scene/pointcloud_sequence.py`, `scene/dataset.py`, `scene/endpoint.py`, dataset timestamps, `scene/noise_calibration.py`.
- `tools/audit_svgd_stein_rejection.py`, `tools/audit_temporal_dynamic_correspondence.py` (`ground_truth`, `verified_labels`) and `tools/evaluate_projective_gt.py`. **Use `ground_truth` only in a subprocess/post-hoc module, never from production.**
- The **finished** run artifacts, if present:
  - `.../svgd_blade_stein_audit_none_20261103_1808`
  - `.../svgd_blade_stein_audit_conditional_temporal_lift_20261103_1809`
  - `.../svgd_blade_stein_audit_none_20261103_1814`
  - `.../svgd_blade_stein_audit_conditional_temporal_lift_20261103_1816`
  - the `svgd_cuda_profile_20261103_1809.json` profile.

`...` means the user-configured results root, *not* a repository path. The results root previously used includes a space (`/media/samuelemara/Extreme SSD/...`); quote full paths. Do not assume these files exist in a different machine/session: inventory first. Recompute only what cannot be reconstructed from immutable artifacts, with a fresh output folder.

## 3. Gate F0: verify observation-model comparability and GT oracle validity

**First deliverable: a provenance table.** Verify source frame IDs, calibration, world frame, timestamp sequence, unique canonical source IDs, full point counts per camera/frame, hard foreground masks, encoded RGB units, and exact values/hashes of `Lambda_X`, `Lambda_C`, `H_X`, `H_C`, RFF `B_X/B_C`, `D`, RFF seed, `Gamma_t` and `Q` used by each run. State whether they are measured, proposed or unauthorized.

The offline GT oracle may use known part membership and joint transforms to build the *physical trajectory of the same persistent canonical points*: `X_GT[0]=X0`, `X_GT[t] = GT_T(t) X0` on verified moving members, `X_GT[t]=X0` on verified stationary members. Validate orientation and world frame, number of verified labels, association-quality distribution, and whether the GT transform is defined relative to frame 0. **If verification fails, stop and report `GT_TRAJECTORY_UNVERIFIED`; do not fake a perfect-motion reference.** Keep the GT construction isolated from estimation.

Do not confuse synthetic depth *measurement* variance with representation bandwidth. Explicitly verify `sigma=q/sqrt(12)` and that source uncertainty is entered exactly once. Report whether the same reference image/points are used in both initialization and the evaluated observation NLL; flag this as conditioned/in-sample evidence. Keep model semantics identical across candidate trajectories.

## 4. Gate F1: is the correct motion actually preferred by the existing observation likelihood?

**Mandatory. This gate takes precedence over any new solver idea.** At the same fixed dataset-calibrated parameters, evaluate the **unchanged normalized `FrameFeatureTerm.nll` used in production** for the following full-sequence trajectories:

- `S`: every persistent canonical particle stays at `X0` for all frames.
- `GT`: verified physical prismatic trajectory, GT used offline only; frame 0 identical to `S`.
- `I(alpha)`: `X_alpha(t)=X0+alpha [X_GT(t)-X0]` with alpha in `{0, .01, .05, .1, .25, .5, .75, 1, 1.25}`. This is a scalar oracle *path through position space*, not an inferred trajectory; for rotation fixtures it is a diagnostic chord interpolation, not a claimed valid rigid path.
- `Stein(eta)`: unaccepted original proposal from saved `Phi`, both original positions-only and optional conditional-velocity-lift variants at **every existing line-search eta**. Confirm reconstructed values match `objective_timeline.csv` to numerical tolerance.
- `GT-motion moving-only`, `static-only` and per-frame composites as controlled **oracle diagnostics** to attribute which support contributes to the feature NLL. These composites never enter inference.

For each candidate produce all `E_obs,t`, sum `E_obs`, absolute and relative `ΔE_obs(candidate,S)` with the **same** RFF features, same `Gamma_t`, same camera/sample set, same normalizers; no refitting or retuning parameters separately per candidate. Record the observation NLL difference from the stationary reference. Global likelihood decomposition is whatever is *actually implemented*; a separate camera-stratified score is an **additional non-equivalent diagnostic**, never mislabeled as production `E_obs`.

**Necessary decision:**
- If `E_obs(X_GT) >= E_obs(X_S)` (modulo independently measured numerical error), the observation model does **not** prefer real motion. Classify `OBSERVATION_DISCRIMINATION_FAILURE`. **Stop attempts to repair it by adjusting proposal step sizes.**
- If `E_obs(X_GT) < E_obs(X_S)`, proceed to study the full objective/prior. This demonstrates a better global endpoint, **not** an attractive local gradient or evidence-calibrated posterior.

Report the per-frame likelihood difference versus GT displacement; whether the loss landscape is monotonic or has barriers; whether a stationary local minimum exists; whether every frame prefers the GT position; and whether frame 0 yields `E_obs,0(X0)` exactly as expected.

### F1 supplemental sensitivity audit (no production change)

Determine whether the kernel-feature representation is the cause of poor discrimination. Re-evaluate the **same physical positions** using independently declared diagnostic RFF feature scales/dimensions (e.g., current B=118mm, representation-scale block, independent multi-resolution block), always label as a **different observation model** and give exact parameter hashes. For small controlled samples compare finite RFF MMD to exact kernel MMD *as diagnostic only*. Separate feature finite-D approximation error from the choice of broad B. No GT-based tuning: this is an oracle falsification sweep and cannot be used to choose production parameters without a later theory amendment and unsupervised calibration.

## 5. Gate F2: compare the same *full objective*, minimizing velocity conditional on positions

For each `S`, `GT` and `I(alpha)` from Gate F1 calculate

```
E_time^*(X) = min_V E_time([X,V])
E_total^*(X) = E_obs(X) + E_time^*(X) + [same unchanged normalizers]
```

Use the exact conditional velocity minimizer already implemented by `TemporalPrior.velocity_lift`, with the **same source-covariance anchor, timestamps and process Q**. Check its KKT residual and agree with independent dense linear solves on a small fixture. It is not enough to initialize `V` from finite differences and compare arbitrary temporal energies. Record separately:

1. frame-0 anchor `½||x0 - x0bar||²_{P0^-1}`;
2. initial velocity prior and process energy in each interval, including position–velocity process cross-terms;
3. the single shared source covariance and each axis/eigen-direction's Mahalanobis contribution;
4. each `E_obs,t`; and
5. numerical normalizers, constant across candidate configurations where appropriate.

**Necessary decision:**
- `E_obs(GT)<E_obs(S)` **and** `E_total^*(GT)>E_total^*(S)` -> `MOTION_PRIOR_DOMINATES` (or likelihood/motion-model relative calibration insufficient). Quantify per-interval contributions; do not inflate Q to get an arbitrary desired outcome.
- `E_total^*(GT)<E_total^*(S)` -> desired physical solution is globally preferable along this oracle comparison; if solver stays stationary, investigate local geometry and proposal acceptance.
- `E_obs(GT)>=E_obs(S)` -> diagnostic confirmed: likelihood itself is insufficient; **do not call it an optimization-only failure**.

Frame-0 should equal canonical coordinates on the GT oracle; if its anchor energy is nonzero, explain the mismatch. The true moving path can still have a large temporal-prior penalty if the selected motion prior is unrealistic. This comparison uses MAP energies, not a marginal Bayes factor; **never label its winner a calibrated moving probability**.

## 6. Gate F3: why does the Stein direction disagree with the feature likelihood?

At the *same* stationary initialization and saved provisional state, record by frame and by post-hoc support:

- `||Phi||`, `||-grad E_obs||`, `||-grad E_total||`, `cos(Phi,-grad E_obs)`, and `cos(Phi, GT displacement)`; include distributions, not only medians. If a vector has negligible norm, report `UNDEFINED_COSINE` instead of assigning a direction.
- `dot(grad E_obs,Phi)` and `dot(grad E_total,[Phi,0])` by frame; validate by central finite differences `([E(X+hPhi)-E(X-hPhi)]/(2h))` over multiple h values until the truncation/roundoff plateau is visible. The existing runner test checking only `sum(grad*Phi)` is **not** a finite-difference oracle.
- Separate attraction and repulsion, and spatial/colour factors; explicit source camera score mass and effective KDE component count. Compute source-support contributions using the **post-hoc `source_mask` decomposition**, leaving production with `source_mask=None`.
- Normalized field direction/magnitude vs **current** GT motion and next-frame incremental displacement, avoiding misleading cosine to cumulative displacement when a local step should track an incremental path.
- KDE basin versus displacement: track component responsibilities, peak contributing observed positions, likelihood at true position vs original position, and whether correspondence mass transfers to unrelated static structures. Evaluate 0–9 frames; pay special attention to the signs around frames 5–6.
- Explicitly distinguish **target-KDE guidance** from the **RFF feature likelihood gradient** and quantify where they disagree. This is a model mismatch even if the individual operators pass numerical tests.

Result classification: `KDE_BASIN_OR_MISASSOCIATION`, `FEATURE_GRADIENT_MISALIGNMENT`, `CROSS_SUPPORT_CONTAMINATION`, `NUMERICAL_GRADIENT_ERROR`, or mixtures, with measured evidence.

## 7. Gate F4: isolate the frame-0 anchor without changing the objective

Use the *identical* `E_total` and initial state. Construct **offline** proposal arms (no production implementation, no approval):

- `A`: exact current `dX_t=eta Phi_t` for all t; velocity unchanged.
- `B`: same positions and exact conditional velocity lift.
- `C`: set `dX_0=0`, `dX_t=eta Phi_t` for t>=1; no velocity lift.
- `D`: same as C plus the **exact conditional velocity lift**.
- `E`: the directional derivative of `E_obs` and `E_total` along each A–D field, independently finite-difference verified.

Evaluate the entire original backtrack sequence **and** an extended, numerically justified descent search (diagnostic only) for C/D. Log `ΔE_obs`, `ΔE_anchor`, `ΔE_process`, accepted/rejected status, physical displacement medians and p90, gradient-angle and first-order approximations. Do not change any covariance, initial state, objective, data term or support constraint. A/A' identity reconstructions must match the saved prior run results exactly; if not, **audit invalid**.

The expected numerical experiment is whether holding `dX0=0` removes the measured `2.817e9` frame-0 anchor cost and whether the remaining process penalty still prevents useful steps. A first accepted step is **not** success unless its displacement and direction are physically meaningful and improvement persists with further steps.

If the offline C/D arms work, propose (but **do not automatically implement**) a documented optimizer-only DEV-0012. Preserve a separate explicit configuration with defaults unchanged and record that this is not the original all-frame SVGD map.

## 8. Gate F5: sequential continuation and information observability (only after F1–F4)

If and only if the unchanged objective prefers true physical motion, determine whether a short-horizon continuation can keep points within the KDE attraction basin. **Offline counterfactual only**:

- Use previously estimated `t-1` states, *never* GT state as an initial guess.
- Compare stationary initialization vs one-step forward-predicted initialization; use the existing fitted state and model without GT labels.
- Do not introduce arbitrary step limits/gates or freeze a hard set of pixel correspondences. Any changed proposal or initial state must be logged and flagged experimental.
- Retain full-sequence backward revision in the theoretical design; a forward warm start is only an optimization strategy, not a replacement objective.
- Check if the sequential scheme learns actual incremental motion without expanding static points or allowing identity drift; report seeded rigid-motion recoverability **as a post-hoc metric**.

If the original likelihood prefers stationary at GT, skip F5. No amount of warm-start engineering proves the missing observation model.

## 9. Required synthetic and real-data treatment

First execute the decisive gates on the same 10-frame, **full** Blade point cloud and the 2-frame run for exact reconstruction. Then use the existing full synthetic translation and hinge/rotation fixtures (not a 30k inference sample), with enough frames to expose sustained motion. Real data is a **later confirmation** after the synthetic failure is understood; the same equations must apply, with independently calibrated uncertainty.

The dataset-specific measurement calibration audit must list depth variance, pixel-coordinate uncertainty, RGB uncertainty, cross-camera covariance when present, representation scales, feature covariance and process noise separately, with provenance, units and consistency checks. No perfect-depth shortcut, no noise-free branch and no GT-dependent parameter fitting. Depth-contour contamination in real RGB-D is a distinct observation-model issue and must not be allowed to mask the synthetic motion failure.

## 10. Required artifacts and visualization (not optional)

Create the following under `RUN/posthoc/svgd_falsification/`, or equivalent fresh output path when existing runs cannot be modified:

1. `preflight.json`: source hashes, commit/working tree, scene/frame/camera counts, selected reference data and GT verification, all uncertainty and scale parameters.
2. `energy_oracle.csv`: **one row per candidate and frame** with `E_obs`, `E_time^*` components, `E_total^*`, GT disp, candidate kind, alpha, and configuration hash.
3. `line_search_counterfactual.csv`: full A–D arms and backtracks, separate NLL/anchor/process contributions, field direction, accepted status and displacements.
4. `stein_field_audit.csv` plus `stein_field_audit.json`: frame-wise and per-(post-hoc) support statistics, source-support mass and effective-component statistics, gradient finite-difference closure.
5. `covariance_provenance.json`: all source/reference/process/feature covariances, eigenvalues, observed vs assumed calibration, no unexplained floors.
6. `diagnosis.md`: concise numerical causal findings, equations and source lines; categorize failure into observation, prior, proposal and basin/association. Include negative and inconclusive findings, not just successes.
7. `theory_deviation_audit.md`: list existing DEV entries and each *counterfactual* model/solver change; no hidden authorization.
8. Adapt `debug/svgd_support_debug.py` and existing browser pages **for display only** with optional overlays: stationary/GT/counterfactual trajectories (GT clearly labeled POST-HOC), per-frame `E_obs` comparison, 3D Stein vectors, source cross-support influence and 0–9 frame traces. Viewer must not change inference arrays or consume GT in inference. Reuse existing QA and add screenshot assertions.

Visual plots should include: `E_obs(X_alpha)` and `E_total^*(X_alpha)` versus alpha per frame and summed; per-frame true displacement vs Stein cosine; line-search energy decomposition on log step size; and distributions of GT-moving vs GT-static contributions. Use physical units and distinguish normalizers/constant offsets.

## 11. Numerical test contract and reproducibility

Add tests that **fail when implementation bugs are deliberately injected**:

- Exact parity with archived 2-frame/10-frame `objective_timeline.csv` on original A/B arms, to reported precision; no regressions of original inference.
- Frame-0 reference identity (`X_GT[0]==X0` and `E_obs,0` exactly agrees), GT verified membership count, camera/world-coordinate correctness.
- Velocity conditional KKT residual and dense oracle comparison; frame-0 anchor plus interval energies add exactly to `E_time`, including process covariance cross-terms.
- Central finite differences for `dot(grad E_obs,Phi)` and `dot(grad E_total,P)` on a small independently reproducible cloud and one real-data trace; report tolerance vs conditioning.
- `C/D` leave X0 bitwise unchanged and remove **only** the anchor displacement, without altering objective or covariance.
- Candidate comparison invariant to diagnostic display cap and camera-loading order up to documented float64 tolerance.
- GT-leak sentinel: importing `tools/audit_*`/GT from any production `estimator/svgd_*`/runner is forbidden; post-hoc audit cannot write `p_dynamic.npy` or alter original artifacts.
- Timestamps/covariance units, SPD and normalizers preserved; compare the production objective to the audit's reconstructed objective with identical fixed Gamma and feature-seed hash.

Tests must state what they establish. Do not claim a model is discriminative simply because gradient math matches finite differences. Provide exact commands, exit codes, test failures, and output paths. Avoid expensive re-runs when saved immutable data can provide exact equivalent evidence.

## 12. Final decision tree and stop conditions

**Stop and report one of the following before more production implementation:**

| Observation evidence | Whole objective evidence | Direction/proposal evidence | Conclusion / permissible next work |
|---|---|---|---|
| GT no better than stationary | any | any | `OBSERVATION_DISCRIMINATION_FAILURE`; propose revisions to feature/visibility/support likelihood on paper first |
| GT better than stationary | GT worse under minimized E_time | any | `MOTION_PRIOR_OR_RELATIVE_CALIBRATION_FAILURE`; inspect selected process Q, source covariance and prior, not arbitrary inflation |
| GT better than stationary | GT better | full Stein rejected, C/D accepted | `FRAME0_PROPOSAL_INCONSISTENCY`; recommend optional proposal-only DEV-0012, **not** a production fix until physical-motion validation |
| GT better than stationary | GT better | C/D still rejected, poor local alignment | `KDE_OR_FEATURE_DIRECTION_MISMATCH`; identify misassociation/basin and propose objective-consistent alternative, theory review first |
| GT better than stationary | GT better | accepted, physically correct short steps but long-term lost | `CONTINUATION_OR_SUPPORT_COUPLING_FAILURE`; design an offline warm-start and coupled-rigidity study |
| GT preferable and stable motion seeds exist | GT preferable | recovery validated | Recommend next frozen theory amendment for coupled rigid coherence and support posterior, **not** automatic implementation |

**Do not call the stage working merely because an offline counterfactual accepts a step.** Required success evidence is physical motion recovery of a spatially distributed moving core, stationary-point stability, ability to fit a rigid motion to the discovered moving core (using the existing downstream routine in a *post-hoc evaluation*), and no unreported deviations. Track seed precision/coverage and pose observability as separate metrics; if calibrated probabilities do not exist, clearly label any seed candidate as **uncalibrated diagnostic**.

## 13. Final Codex report format

Return a self-contained report with:

```
CHECKOUT_COMMIT / UNCOMMITTED_CHANGES / FILES_EDITED
ORACLE_VERIFICATION (including GT membership reliability)
MODEL_PARAMETERS_AND_UNCERTAINTY_PROVENANCE
E_obs(GT)-E_obs(STATIONARY) [each frame + sum]
E_time^*(GT)-E_time^*(STATIONARY) [anchor / process / velocity]
E_total^*(GT)-E_total^*(STATIONARY)
E_obs and E_total^* versus alpha (table/plots)
ORIGINAL_STEIN_GRADIENT_FD_CLOSURE
COUNTERFACTUAL_A/B/C/D [energy decomposition, acceptance, physical displacements]
KDE_BASIN_AND_SOURCE_CROSS_SUPPORT_AUDIT
ROOT_CAUSE_CLASSIFICATION (measured evidence, not hypothesis alone)
SYNTHETIC_NOISE_AND_REAL_NOISE_SEMANTICS (same model, separate calibrations)
TESTS_AND_INVARIANCES / VISUAL_QA / PERFORMANCE
DEVIATIONS_PROPOSED_NOT_AUTHORIZED
EXACT_NEXT_DECISION_REQUIRED
```

**The task is complete once this diagnosis is proven and documented, even if every experimental proposal still fails. Do not respond by implementing new A1–A7 posterior/visibility models or claiming success from tests that check only algebra or browser rendering.**
