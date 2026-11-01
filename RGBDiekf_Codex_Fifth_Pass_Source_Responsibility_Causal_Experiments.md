# CODEX — FIFTH PASS: SOURCE-CAMERA RESPONSIBILITY, PIXEL–POINT ASSOCIATION, AND CROSS-CAMERA BIAS

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Mode:** **DIAGNOSTICS AND SMALL OFFLINE REPLAYS ONLY. DO NOT MODIFY PRODUCTION INFERENCE.**  
**Priority:** Resolve whether the static-drift initiator is invalid cross-camera pixel-to-point correspondence, a valid observation with a persistent camera-specific RGB-D discrepancy, inappropriate source-point suppression, or a combination. Quantify the consequences for moving points. **Do not preselect the winner.**

## 0. The question to answer

A point `i` is reconstructed at the initial frame from an exact RGB-D observation identified by:

- original point ID `i`;
- source camera `c_src(i)`;
- source pixel `p_src(i) = (u_src, v_src)`;
- original world coordinate `x_i^0` and canonical appearance `a_i`.

In the standard forward F1 IEKF, *every camera* may produce a row for `i` if the point's current projection passes current point-cloud visibility, nearest-pixel ownership, mask, depth, and innovation tests. The point's original source-camera identity is not used to establish material correspondence of subsequently sampled pixels. Thus a geometrically admissible non-source camera is sometimes treated as measuring the exact same 3-D material point even if this was never established.

**Main question:** Is the first false static displacement caused by this loss of source-camera/pixel responsibility, by persistent RGB-D discrepancies *within valid camera-point associations*, by the single-pixel winner rule wrongly suppressing a source observation, or by numerical instability? Which contributes how much, for which points?

Do **not** confuse the following three concepts:

1. `identity(i)`: permanent material-point/canonical endpoint ID;
2. `responsibility(i,c)`: whether camera `c` has a justified observation of that point;
3. `pixel(i,c,t) = pi_c(x_i,t)`: a position that **must be allowed to change** as a point moves.

**Do not freeze the pixel coordinate at `(u_src,v_src)` through time.** That would prevent real motion. The candidate idea is to preserve observation *identity/provenance* and test camera eligibility, not prohibit reprojection.

**Do not infer that the original RigEuler3DGS permanently fixed frame-0 pixel assignments.** Its frozen linearization keeps renderer contributor lists and Jacobians fixed during one linear solve, but re-renders/re-sorts after accepted nonlinear steps. Verify exact code and report what it did, rather than describing it as permanent frame-0 pixel ownership. RGBDiekf and RigEuler3DGS also differ in representation and whole-image versus per-point likelihood.

### Known evidence — preserve the corrections

- Storage `8444`: original source camera **0**, but its frame-1 source row is `DUPLICATE_PIXEL_SUPPRESSED`; **camera 1 alone** drives approximately **24.156 mm** initial displacement and ~**1.411 m/s** injected velocity. Camera 1's RGB mismatch exists unchanged from frame 0 onward. RGB-D frame-0 referencing suppresses drift over frames 0–10 in the previous *isolated point* replay.
- USB `67827`: original source camera **5**, admitted camera **6** has about **-41σ**, **-2.28 mm** depth discrepancy across a crease already present at frame 0. Camera **4's -80σ row was NOT used at frame 1**; it can enter later because it lacks a valid frame-0 reference. Depth referencing reduced static drift in the fourth-pass short replay.
- USB moving `87290`: current F1 loses admitted rows when its prediction drifts; later replays still fail to follow true motion. At first loss, some camera rows fail χ² even though the ground-truth position is visible. The latter is **evaluation only**, not legal tracking information.
- USB moving `85696`: all camera rows become duplicate-pixel suppressed at frames 3–5, including under referenced variants; source-only policy could be even worse.
- Storage moving `8416`: micro-replay differs from production at frame 1 due to non-smooth pixel-boundary behavior. Treat **all quantitative intervention comparisons for this ID as INDETERMINATE until parity is repaired**. Earlier claims of 0.65 m etc. were *mixed position/velocity units* and are retracted. Position/velocity differences must be separately labeled.
- Fourth pass: baseline replay matches production continuously for **seven of eight** mandatory cases, down to ~1e-12 m position; all 17 tests passed. Preserve this baseline and its hashes.

**Previous 'all cameras sample the point' is an oversimplification:** production does *not* use every camera for every update. It projects all points into all cameras but accepts only a subset after nearest-pixel ownership and other gates. Inspect **actual `in_accepted_solve` and frozen ownership**, not merely a camera sample being valid at projected coordinates.

## 1. Source-of-truth code and inputs

Read **current local** code; the GitHub public branch is not guaranteed to contain the newest diagnostic tools. Specifically:

- `scene/endpoint.py`: `endpoint_samples`, back-projection, canonical RGB, full source indices.
- `estimator/free_projective_run.py`: `build_catalogue`, canonical `original_source_ids`, initial state/covariance, `run_free` and the standard F1 loop.
- `estimator/projective_camera.py`: world/camera conventions, project/backproject, camera matrices.
- `estimator/projective_sensor.py`: exact bilinear sample, four-pixel mask constraint, `same_surface` check, `R`.
- `estimator/projective_terms.py`: actual residual, Jacobian, row innovation/gate.
- `estimator/projective_visibility.py`: `visibility`, frontmost pixel winner, `DUPLICATE_PIXEL_SUPPRESSED`, `SELF_OCCLUDED`, `apply_frozen_assignment`.
- `estimator/point_kalman_cuda.py`: `PointMeasurement`, frozen per-frame assignment, exact `update_points`, candidate acceptance and posterior covariance.
- `tools/root_cause_oracle.py`, `tools/f1_second_pass.py`, `tools/f1_temporal_reference_diagnostics.py`, `tools/f1_sequential_causal_replay.py`, `tools/f1_diagnostic_units.py`: **reuse**, do not re-implement unnecessarily.
- `RigEuler3DGS/estimator/renderer_bridge.py`, `RigEuler3DGS/estimator/sparse_iekf.py`, `RigEuler3DGS/submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.h`: document what contributor freezing actually means; this is context, not permission to port the renderer.

Existing run root (do not overwrite):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
FOURTH="$FOR/fourth_pass_sequential_replay_20261030"
SECOND="$FOR/second_pass_20261030"
THIRD="$FOR/third_pass_temporal_reference_20261030"
OUT="$FOR/fifth_pass_source_responsibility_20261030"
```

**Use the exact saved run manifests, frame IDs, calibration, `x0`, `original_source_ids`, source image sizes, original F1 `P0`, `Q`, `R`, timestamps, camera reason/status/point IDs, and fourth-pass reference validity.** No assumed intrinsics, default-camera counts, made-up thresholds or resampled observations. Check hashes and record provenance before running.

The previous fourth-pass report includes these output columns (reuse/parity-check):

- `sequence_state.csv`: `case, arm, mode, frame, x_minus_m, v_minus_m_s, x_plus_m, v_plus_m_s, observed, acceptance, rows_initial, rows_accepted, ...`;
- `sequence_camera_rows.csv`: `admitted_at_s_minus`, `in_accepted_solve`, `frozen_or_recomputed_code`, original `r_model`, `R_model`, `nis_model`, `reference_applied`, etc.;
- `ownership_mode_C_parity.csv`: recomputed all-cloud pixel ownership versus saved production;
- `sequence_objective.csv`, `sequence_failure_events.csv`, `reference_covariance_tests.csv`, `FOURTH_PASS_CAUSAL_REPORT.md`, `tests.log`.

**Filesystem rule:** create a new output directory only. Do not overwrite the original diagnostics, production outputs or existing reports. Do not commit code. No GPU or long full-scene IEKF run.

## 2. Implement the provenance/responsibility auditor first

New diagnostic tool suggestion: `tools/f1_source_responsibility_audit.py`. Design reusable callable functions, not a monolithic print script.

### 2.1 Exact frame-0 map for EVERY selected point and camera

For each point `i` and camera `c` record:

- original `source_frame`, `source_camera`, `source_u`, `source_v`, original full endpoint index, canonical XYZ/RGB;
- projection at `x_i^0`: `(u0_c,v0_c,Z0_c)` and source reprojection discrepancy in pixels, with exact `float64`;
- 2x2 *raw* depth/RGB/mask pixels, the four bilinear weights, interpolated residual and gradients, `R`;
- whether point `i` is truly in front of the frame-0 endpoint **point cloud** from that camera, using the exact production visibility algorithm; its nearest-pixel owner PID and **that owner's source camera/pixel**, and world/depth separation;
- whether camera `c` images a single locally consistent surface or a depth crease/discontinuity (report depth spread and second differences, no unvalidated binary 'same material' claim);
- mask validity, self-occlusion, duplicate suppression, source/non-source designation, row gate at `x0`;
- frame-0 residual `b_i,c,0 = h_i,c,0(x0)` and explicit validity or failure reasons.

**Important:** a frame-0 non-source camera that projects onto the foreground is only a **candidate** association. Depth agreement does NOT establish the same *material point*, especially along a textureless plane. Record `FRAME0_GEOMETRICALLY_PLAUSIBLE`, `FRAME0_INCONSISTENT`, or `UNDECIDABLE`; do not call `VERIFIED_MATERIAL_ID` without genuine material evidence.

Source-camera association is known at frame 0 by construction, but the source point may lose a nearest-pixel winner contest due to additional overlapping canonical points. Do not confuse `the source pixel measured the point` with `the current raster ownership allows the point to be updated`.

### 2.2 Frame-1 / first-update actual responsibility

For every included point, reconstruct **exact** per-camera status at prior, each IEKF iterate and accepted state (if available):

- `candidate_at_projection`, `production_frozen_owner`, `sample_valid`, `row_gate_passed`, `used_in_information`, `accepted_state_row`;
- actual competing owner PID (full endpoint ID, source camera/pixel, projection XYZ/depth, predicted covariance); distinguish ties from true occluders;
- `source_camera_suppressed` (exact cause) and whether another camera becomes the only admitted row;
- `r_c,0(x0)` vs `r_c,1(x0)` and `Delta r`, distinguishing genuinely *unchanged* residuals from temporal changes;
- actual measurement drive for each camera/channel `g_c = H_c^T R_c^-1 r_c` **and the complete-state counterfactual solve with the other rows unchanged**; do not interpret `||g_c||` alone as its positional contribution;
- per-row `S`, `NIS`, `R`, `H`, and accepted `delta x`, `delta v`, convergence, cost with comparable measurement support.

Critically, **compare source, cross-camera, and actual admitted subsets**, not all projected cameras. Reproduce corrected second-pass erratum: Storage 8444 only camera 1 used, USB 67827 cameras 5 and 6, and exclude false occluder claims.

### 2.3 A compact responsibility matrix

For selected IDs output one table per frame with cameras as columns and codes:

- `SRC_VALID`, `SRC_SUPPRESSED_BY_POINT:j`, `CROSS_VALID_REF`, `CROSS_VALID_NO_REF`, `CROSS_DEPTH_CREASE`, `CROSS_MASK_INVALID`, `CROSS_OCCLUDED`, `CROSS_DUPLICATE`, `CROSS_CHI2_REJECT`, `UNDECIDABLE`.

These are **debug classifications**, not hardcoded inference states. Always also write raw status flags to avoid merging multiple rejection causes.

## 3. The controlled experiments — full 2×2 association × bias factorial

The central experiment must be **factorial**. Merely switching to source-only and seeing no drift cannot identify whether the non-source row was physically wrong or just had a persistent but correctable bias.

Use the *same saved prediction prior*, full covariance, original solver, accepted-state rule and frame-1 images for a first-frame comparison. Then run independent **continuous frame-0-to-10** single-point micro-replays for the same variants (each accepted posterior propagates). Freeze the full-cloud competitor trajectories from the original saved F1 output for ownership; record that this is a **conditional one-point counterfactual**, not a coupled all-point rerun.

### 3.1 The six minimum arms

| Arm | Camera eligibility (before normal sampling/gates) | Measurement residual | Purpose |
|---|---|---|---|
| `A_ALL_RAW` | Original F1 admitted cameras | Original RGB-D | Exact production baseline |
| `B_SOURCE_RAW` | Only `c_src(i)`, if eligible under **original ownership** | Original | Does removing non-source rows remove the false correction? |
| `C_FRAME0_PLAUSIBLE_RAW` | Source + geometrically plausible and physically visible frame-0 non-source cameras | Original | Does frame-0 camera provenance improve association? |
| `D_ALL_REFERENCED` | Same cameras as A | Camera-specific valid frame-0 residual subtraction, with declared variance mode | Bias correction without eligibility change |
| `E_FRAME0_PLAUSIBLE_REFERENCED` | Same cameras as C | Frame-0 reference on eligible cameras; source reference handled correctly | Combined association and bias |
| `F_SOURCE_REFERENCED` | Same cameras as B | Source-camera reference when valid | Factorial control: referencing cannot silently create an observation |

`C` and `E` must use **only frame-0 measured geometry, masks, cameras, and stated uncertainty**. Do not use GT labels or final trajectory to decide eligibility. Where material identity cannot be established, retain explicit `UNDECIDABLE`; run two *labeled sensitivity interpretations* (admit vs exclude ambiguous rows) rather than silently claiming certification.

For missing valid frame-0 references, never set `b=0` by pretending that is a measured reference, and never import another point's or camera's reference. Report `NO_VALID_REFERENCE` and either retain the original residual or omit the row **as separate explicit arms**; these choices have different interpretations. In particular USB 67827 camera 4 is a mandatory test when it enters later.

Use the same `R` in the pure eligibility comparisons. For referenced residuals report both existing diagnostic `ALGEBRA_ONLY` and `R_t+R_0` as *sensitivity*, NOT an exact multi-frame Kalman covariance. Shared frame-0 noise creates cross-time correlations; **do not claim the referenced sequential posterior covariance is statistically exact**. For a single-frame comparison, retain original `R` when isolating mean-response changes, with an explicit non-probabilistic label.

### 3.2 Ownership controlled separately (do not conflate with eligibility)

Repeat selected arms under two ownership modes:

- `P_PRODUCTION_OWNERSHIP`: exact original frozen per-frame assignment (must match baseline, including suppression of original source camera);
- `C_RECOMPUTED_OWNERSHIP`: recompute the current point's competition against frozen saved trajectories of **all other points**, exactly as the fourth-pass mode C; prove parity for `A_ALL_RAW`.

And two **source-ownership probes**, strictly diagnostic:

- `SRC_OBSERVATION_PROBE`: *evaluate* the source-camera row at the current projected coordinate even when it is duplicate-suppressed, without necessarily adding it to the solve; report image validity, depth, whether the current winner could physically occlude it, and original source pixel provenance.
- `SRC_PRIORITY_COUNTERFACTUAL`: change only the duplicate tie winner for a source-camera point, **when the production geometry says duplicate rather than genuinely self-occluded**. Re-run the complete one-point update with this alternative assignment, and flag the fact that a multi-point global ownership change would transfer the same pixel's measurement from the previous winner. This is an ablation, not a legal production fix or proof of point visibility.

A winning source identity **must not** override mask invalidity, out-of-frustum, measured-depth occlusion or established frontmost surface physics. Do not count the same RGB-D pixel as independent evidence for two different points. If an arm evaluates both anyway for illustration, label `INVALID_DOUBLE_COUNT_COUNTERFACTUAL` and exclude it from claimed posterior quality.

### 3.3 Source-camera-only cannot be assumed to solve moving-point tracking

All arms must use the current projected pixels of transported points, not source pixel locations fixed in the original image. A moving point can be occluded or leave a view. Source-only may remove essential complementary camera geometry and cannot be considered successful just because it holds a static point still.

For moving point 87290 and 85696 report accepted rows, true-positive *image evidence* available without GT, and endpoint GT error **only in a disconnected evaluator**. A zero-motion solution with no rows is failure on moving controls.

## 4. Mandatory point IDs and causal hypotheses

**Core:**

1. `storage_45135 / 8444` (static, source 0, erroneous camera 1): detailed source-owner conflict at frame 1, complete six-arm factorial and source-priority probe, frames 0–10.
2. `USB_100109 / 67827` (static, source 5, erroneous camera 6, late camera 4): isolate original camera 6 and missing-reference camera 4, frames 0–10. Never claim camera 4 caused the frame-1 update.
3. `USB_100109 / 87290` (moving): frame 1 through first all-camera gate loss, continue to frame 10; compare source-only / candidate reference / all-camera; exact NIS + accepted-solve status.
4. `USB_100109 / 85696` (moving): the same, with first duplicate-suppression cascade; actual source winner and competing owner identity.
5. `storage_45135 / 8416` (moving, pixel knife edge): record status and owner but **quantitative ablation verdict INDETERMINATE** unless the baseline single-point oracle reproduces production to the exact parity tolerance. Never silently remove the point.

**Controls and counterexamples:**

- `storage_45135 / 49199`, `262894`, `60019`, `100564`;
- `USB_100109 / 30502`, `51975`, **neighbor `51974`**, optionally `30503`;
- `blade_103706 / 10251`, `14104` where baseline traces are available.

Select extra high/medium/low-drift static IDs **deterministically before knowing their ablation response**, stratified by source-camera number, non-source admitted-row count, frame-0 residual magnitude, depth slope, and whether source row was duplicate-suppressed. Record selection seed and all excluded points. Include cases with no non-source rows and with plausible multiview support. Do not choose only cases supporting the hypothesis.

## 5. The indispensable source-camera winner audit

For Storage 8444 camera 0 frame 1, and USB 85696 across frames 2–5, record the **actual source-pixel competition**:

```text
point_i, its_source_camera, its_source_pixel, its_predicted_projected_pixel,
owner_point_j, owner_j_source_camera, owner_j_source_pixel,
Z_i, Z_j, Z_i-Z_j, predicted_sigma_Z_i, predicted_sigma_Z_j,
mask/depth sampling validity, owner_is_true_front_surface_or_near_tie,
production_reason, source_row_residual_if_evaluated,
source_row_effect_if_only_pixel_winner_changed,
other_point_j_row_lost_in_counterfactual, double_count_flag
```

Check **both** the exact round-to-nearest integer ownership pixel and the 2x2 bilinear sampling footprint: a point may win one integer pixel yet interpolate across several surface footprints. Produce a 15x15 or similarly small per-camera image crop with point `i`, owner `j`, source pixel and current projection marked; no large video render.

At frame 0, if two points were reconstructed from the same measured ray or overlapping ray footprints, identify whether the catalog already duplicates surface samples. **Source provenance alone does not guarantee that the point remains the correct winner after motion**; if the same pixel has multiple canonical point IDs, report this ambiguity rather than artificially selecting one by point ID.

This is the exact test of the user hypothesis: **does an original source camera lose the right to update its own point merely because of per-frame nearest-pixel selection among points from all cameras?**

## 6. Independently differentiate wrong correspondence from valid but biased observation

Use the same point, same camera, same prior, and same other rows for three orthogonal tests:

**T_bias (fixed camera and support):** subtract only that camera's *valid* frame-0 reference discrepancy from the current residual, keeping camera eligibility and its pixel footprint unchanged at the evaluated state. Compare position and velocity update, nonlinear accepted cost and per-camera information. If correction disappears and source geometry is consistent, that is evidence for a persistent cross-view observation bias **but not proof of material-point correspondence**.

**T_assoc (fixed residual model):** remove that camera's row, holding the rest and priors constant. If the correction disappears, the camera caused it; this alone **does not distinguish** a wrong surface from a biased valid observation. Evaluate whether the camera's depth neighborhood at frame 0 and at the first accepted state could represent the same geometric surface; output `WRONG_SURFACE_SUPPORTED`, `CONSISTENT_SURFACE_PLAUSIBLE`, or `MATERIAL_ID_UNDECIDABLE`. Do not infer material ID from part-level GT labels (these may have coarse point spacing).

**T_owner (fixed camera/residual, swapped winner):** restore a suppressed source camera's row only when raw camera/mask/depth evidence is valid and the collision is a production `DUPLICATE`, not a true physical occlusion. Identify precisely which competing PID loses ownership in the full-coupled interpretation. Compare inferred motion; **do not score this as a valid independent multi-point posterior**.

**T_nonlinear:** for each decisive update evaluate original-row-set likelihood at prediction and accepted position wherever physically valid; separately report candidate-row likelihood. Record cell changes, convergence and accepted iteration. If rescue appears only via a nonconverged/unstable iterate, mark mechanism `NUMERICALLY_INDETERMINATE`.

**T_temporal:** compare r(c,t,x_ref) to r(c,0,x_ref) for true-static ground truth *in post-hoc evaluation only*. A temporally constant discrepancy is evidence of no new temporal signal, not automatic proof that its cross-view row is a wrong material match.

A causal interpretation requires agreement across **all applicable** tests, not just a single lower endpoint drift.

## 7. Population-level check (with the existing saved F1 result)

Only after the point-level oracles pass, do a **read-only, chunked** audit of the existing original endpoint cloud and saved standard-F1 arrays for USB, Storage and optionally Blade. Use frame-0 and frame-1 raw images and saved `x0`/`x_minus`, without re-filtering the scene.

For every physically grounded static point where GT labels are available *for scoring only*, report:

- source camera/provenance and whether source camera row was actually in the first accepted update;
- count of non-source admitted rows; count of non-source rows with invalid/unavailable frame-0 reference;
- magnitudes of original persistent `r(c,0,x0)` and change `r(c,1,x0)-r(c,0,x0)` (RGB separately from depth);
- first accepted position and velocity corrections, first nonzero velocity and drift onset;
- whether the non-source row dominates the *actual* 6D normal-equation correction (whole-Hessian counterfactual, not just `z_D`);
- whether the source camera was suppressed, by what competing original point ID, and whether that winner was also source-camera originated;
- whether the point's source row was physically valid if ownership alone is removed;
- static-error strata from GT **only after** all eligibility and hypothesis variables have been computed.

Construct *conditional* risk tables with denominators shown, e.g.:

| First-update category | N | Median Δx | Median Δv | Fraction drift >1 cm at frame 10 | Fraction drift >10 cm final |
|---|---:|---:|---:|---:|---:|
| source row admitted, no non-source rows | | | | | |
| source admitted + non-source rows, no frame-0 mismatch | | | | | |
| source admitted + persistent non-source mismatch | | | | | |
| source suppressed + non-source row admitted | | | | | |
| source suppressed + no admitted rows | | | | | |
| non-source camera without valid frame-0 reference | | | | | |

Do not infer causality from these associations alone. Compare them with the individual intervention tests. Report both entire population and strata by **number of admitted cameras, depth/texture regime, initialization covariance, source camera, and source-row eligibility** to avoid confounding.

Avoid an implicit massive all-frame `N × C × T` memory tensor. Compute in chunks, and save summaries and selected event rows only. If this audit is expensive, perform frames 0–1 plus the selected frame-10/endpoints; do not launch a new standard F1 full-scene pass.

## 8. Replaying the real update — fidelity requirements

The tool must produce an **exact baseline replay** `A_ALL_RAW` for each ID and each frame 1–10 with:

- per-point 6-state `[x(m), v(m/s)]` and full 6x6 covariance;
- actual `F`, `Q`, `dt`, `P0`, `R`, canonical RGB;
- pixel ownership at predicted state frozen per frame as production specifies;
- all camera row gates, mask, raw-depth validity and allowed discontinuity rules;
- the same 10 iterations, convergence and `last_consistent_iterate` acceptance;
- physically identical treatment of invalid rows and Joseph/information covariance;
- the original full-cloud saved trajectories used only as **frozen competitors** in source/no-source conditional replay (all PIDs available; do not rebuild source associations from a partial subset).

**Parity gate:** for seven previously reproducible mandatory points, `A_ALL_RAW` must reproduce the fourth-pass saved state and production up to existing documented tight tolerances (at most `1e-10 m` position, `1e-9 m/s` velocity, and identical camera row codes and accept/reject flags); report actual max errors rather than merely print PASS. Reproduce baseline `P+` to the prior test's relative tolerance. If parity fails, **stop that point's causal claims** and debug the replay. Storage 8416 remains indeterminate until resolved.

A micro-replay of a point under modified ownership is **not** an exact coupled multi-point simulation; competitor trajectories remain fixed, and swapping the winning point would alter that competitor's update in reality. Report `CONDITIONAL_FROZEN_COMPETITORS` prominently. Do not claim validated multi-point tracking or global cost improvement.

When comparing model costs, compare **identical observation sets** and exact model definitions before/after; report invalid comparison if rows become impossible rather than treating missing rows as zero error. Do not use unrelated models' raw objective magnitudes as if directly comparable probability scores.

Record separately `position_change_l2_m`, `position_change_linf_m`, `velocity_change_l2_m_s`, `velocity_change_linf_m_s`. **Never take `max()` over the mixed six-state and label it meters.**

## 9. Required evaluation and success/failure tests

For each arm over frames 0–10 report:

- full per-frame accepted and predicted position, velocity, covariance block eigenvalues;
- number and actual IDs of admitted source/non-source camera rows;
- exact first false correction and velocity injection (static points), accepted iteration and nonlinear convergence;
- first loss of source row and first loss of all rows, with competing winner PID and actual reason;
- canonical drift for static points, GT motion-tracking error for moving points **only as eval**;
- difference against `A_ALL_RAW` and against the corresponding reference-only or source-only arm;
- counterfactual interpretation, including visibility, reference uncertainty and possible double count;
- whether corrected static behavior is obtained at the cost of losing all moving-point observations.

**Mandatory tests** (create `tests/test_f1_source_responsibility.py`, use CPU float64 and small fixtures):

1. Canonical `original_source_ids` exactly maps camera ID and integer pixel to the source back-projection; source reference residual zero at frame 0 by construction (with acknowledged prior correlation).
2. A point retains the same ID while its projected pixel moves under nonzero physical motion; fixed frame-0 pixel policy must be explicitly rejected.
3. `A_ALL_RAW` reproduces saved baseline state, covariance, rows and acceptance on a stable ID and on failing 8444 and 67827.
4. `B_SOURCE_RAW` is precisely `A_ALL_RAW` with non-source rows excluded; no accidental changes to `R`, `P`, `Q`, gate, or acceptance.
5. Source-observation probe does not force valid status for physical occlusion, mask exclusion, invalid depth or outside FOV.
6. Duplicate competition is reported with both original point IDs and source-camera/pixel provenance. Source-priority changes the **winner**, not the number of independent sensor observations in a physically valid assignment.
7. A camera with an unavailable frame-0 reference is never silently assigned a zero reference; USB camera 4 is a real regression case.
8. Non-source frame-0 depth/visibility agreement does not automatically assert material point identity on a uniform planar surface.
9. RGB-only versus depth-only reference factors isolate Storage 8444 and USB 67827 as previously established; if not, block the new verdicts.
10. Point 51974 stays stable, but moving points 87290 and 85696 must not be declared successful merely because their velocity stays small.
11. A physically valid source row excluded solely by duplicate ownership is measurable in *probe* mode, but is not double counted in any reported valid estimation posterior.
12. If a camera support becomes physically invalid after the proposed move, the comparable-row objective is flagged `UNDEFINED_SUPPORT`, not zero-filled.
13. Finite-difference and exact normal-equation parity checks for changes in camera-row inclusion; channel contributions sum to the full correction under a common Hessian.
14. No GT data is imported or used by policy selection, update, reference construction, or eligibility; GT is loaded only in the post-hoc evaluator.
15. Determinism and read-only invariance: repeated diagnostic runs give matching outputs/hashes, and no production file/config/existing result is modified.
16. Explicit 8416 knife-edge test demonstrates position-vs-velocity units separately and marks parity-failing comparisons indeterminate.
17. Synthetic toy case: two cameras see different material patches with identical depth/foreground mask; neither pure geometry nor `D` agreement can determine material identity. Report `UNDECIDABLE` rather than a fabricated match.
18. Synthetic toy case: a true moving point has a changing projection in its source camera and becomes occluded; a source-only policy is correctly not forced to use that camera.

No acceptance based on mere `unittest ... | tail -4`; save the complete tests log and process exit code. If baseline parity or any required unit test fails, diagnose and correct the *diagnostic*, not production inference.

## 10. Minimum output artifacts

New output root `$OUT` (never overwrite existing):

```text
README.md                               exact commands, data paths, baseline hashes, CPU-only note
responsibility_frame0.csv              one point × camera, source ID/pixel, physical validity, owner PID/source, residuals
first_update_responsibility.csv          prior/iterate/accepted row & source/cross classification, owner, camera/channel contribution
source_owner_conflicts.csv              every source-camera suppression with winner identities, raw depths/visibility
association_counterfactuals.csv          A–F arms, original and source-priority controls, frame-level acceptance/costs
sequence_state.csv                      frames 0–10: complete mean/cov summaries for each arm, metric units
sequence_rows.csv                       actual per-camera row admission and rejection reasons for each arm/frame
sequential_failure_events.csv           first false correction, first source loss, first all-camera loss, reacquisition
population_source_vs_cross.csv          chunked production-aligned strata with denominators
causal_evidence.json                    point-by-point competing hypotheses and supported / contradicted / undecidable
rig_euler_vs_rgbdiekf_correspondence.md verified code-map and conceptual difference; no falsely claimed permanent pixels
figures/                               small 2D/3D pixel-owner overlays, trajectory & per-camera information plots
tests.log                               full test output + exit code
SOURCE_RESPONSIBILITY_CAUSAL_REPORT.md  final controlled-results report
```

Keep CSV column headers and units explicit. Every figure must state which point, camera, frame, source origin, winner PID, residual model and arm it depicts. Limit diagnostic snapshots to decisive points/frames (no bulk image/video dump).

### Final report — MUST answer these questions explicitly

**For Storage 8444:**

- Does source camera 0 actually contain a physically valid row at frame 1 but lose ownership? Which exact point wins and where did **that** point originate?
- What changes when camera 1 is removed, when camera 1 is frame-0 referenced, when camera 0 alone is probed/restored, and when eligibility and bias corrections are combined?
- Which of these interventions cures the first 24.156 mm displacement? Which remains effective across 10 frames? How much is dependent on a potentially invalid double-counting/source-priority counterfactual?
- Is camera 1 a wrong physical surface match, a correct material observation with persistent RGB bias, or not identifiable from these images?

**For USB 67827:**

- Is camera 6 an actually justified material-point observation, or simply a foreground crease pixel? Is its 2x2 depth support geometrically self-consistent?
- How does retaining camera 5 source and removing/reference-correcting camera 6 change the first update?
- Which camera and which explicit responsibility transition cause the later camera-4 jump, given missing frame-0 reference?

**For moving 87290 / 85696:**

- Does retaining the source-camera association preserve enough *moving evidence* to track, or does source-only make moving tracking worse?
- Is first all-camera loss due to false prior motion, chi2 rejection, ownership, actual surface visibility, or no identifiability? Give camera-by-camera proof, not a summary label.
- Where is the information about the correct motion present in a *non-GT* image observation, and why does the current association rule fail to use it?

**Population:** Is it common among severe static drifters that the source camera is suppressed while a non-source camera with an already inconsistent frame-0 residual drives the update? Quantify N, prevalence, risk conditioned on available camera count and uncertainty; identify counterexamples.

**Architecture diagnosis:** Which *specific function and logic* (file + line) first replaces known source provenance with unvalidated pixel-point responsibility? Identify separately the first physical observation-model failure, the position→velocity amplifier, the first subsequent row-loss mechanism, and any optimizer instability. No generic 'wrong association' verdict without point/camera/pixel numerical proof.

**Decision table:** `ASSOCIATION_PRIMARY`, `BIAS_PRIMARY`, `OWNERSHIP_PRIMARY`, `MIXED`, `OTHER`, `INDETERMINATE` for each point. Include evidence against the leading hypothesis. If a surface cannot be uniquely identified, mark `MATERIAL_ID_UNDECIDABLE` even when a counterfactual eliminates drift.

## 11. If the hypothesis is confirmed: candidate *design*, not unauthorized inference code

After all evidence and tests pass, Codex may write `CANDIDATE_RESPONSIBILITY_DESIGN.md` explaining the smallest plausible change. It may **NOT modify production** in this pass.

A candidate design should preserve:

- permanent canonical point ID and source observation provenance;
- evolving projected pixel coordinates under motion;
- explicit camera–point association uncertainty and visibility; non-source cameras may become valid later, but are not assumed to own the same material point merely because they project onto foreground depth;
- source camera as provenance anchor, **not** an unconditional visible row;
- properly modeled camera-specific persistent appearance/depth discrepancy, if supported, including shared-reference correlations;
- masked foreground-only measurement policy;
- no arbitrary velocity gates, no GT-based association, no reintroduction of full 3DGS rasterization or slow all-pixel dense rendering;
- efficient sparse GPU-compatible observation/indexing for future implementation, while this pass remains CPU diagnostic;
- correct avoidance of double-counting shared sensor pixels across competing points;
- nonlinear acceptance/convergence and pixel-boundary Jacobian stability as independent unresolved issues.

**A proposed weighted information sum is not automatically a valid Bayesian filter** when point responsibilities are random or measurements are shared. Specify a valid likelihood/latent association interpretation or explicitly call it a weighted least-squares approximation, and state what uncertainty is omitted.

## 12. Execution policy and stop rule

1. Inspect local code and current archive; print resolved paths, config and source hashes.
2. Implement the read-only responsibility map; **prove source ID ↔ camera/pixel provenance with source reprojection test**.
3. Obtain `A_ALL_RAW` baseline parity; **block claims for any point that fails**.
4. Run point-level frame-1 association × reference factorial, then short sequential 0–10 arms.
5. Run source-winner microprobes and moving-point controls.
6. Run chunked frame-0/1 static-population observational audit (only if no full-scene filter is necessary).
7. Run all tests, render small decisive crops, produce the evidence ranking and design *only if supported*.
8. Do not commit; leave `git status --short` and SHA256 manifest in the report.

Run under `CUDA_VISIBLE_DEVICES=""`; consult `--help` for the exact implemented CLI; examples below are **desired new CLI contracts** Codex should implement, not statements about currently existing flags:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
OUT="$FOR/fifth_pass_source_responsibility_20261030"
CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/rigeuler3dgs/bin/python \
  tools/f1_source_responsibility_audit.py \
  --forensics-root "$FOR" \
  --fourth-pass-root "$FOR/fourth_pass_sequential_replay_20261030" \
  --out "$OUT" \
  --frames 0:10 \
  --experiment all \
  > "$FOR/fifth_pass_source_responsibility_20261030.log" 2>&1
RC=$?
printf 'exit_code=%s\n' "$RC" >> "$FOR/fifth_pass_source_responsibility_20261030.log"
[ "$RC" -eq 0 ] || exit "$RC"
CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/rigeuler3dgs/bin/python -m unittest \
  tests.test_f1_source_responsibility -v \
  > "$OUT/tests.log" 2>&1
RC=$?
printf '\nexit_code=%s\n' "$RC" >> "$OUT/tests.log"
cat "$OUT/tests.log"
exit "$RC"
```

### Nonnegotiable stopping condition

**Do not declare the problem solved because a source-only replay is static or because reference subtraction zeroes unchanged images.** The pass is successful only when the exact first incorrect update is causally attributed to specific admissible/non-admissible point-camera-pixel associations, the bias/ownership confounders are separated, the explanation is reproducible against production, and moving-point controls show whether the proposed responsibility policy preserves or destroys real motion evidence. If the observations are insufficient to determine material identity, state that fact and specify the missing evidence.

The demanded deliverable is an **unbiased, reproducible decision about the original responsibility hypothesis**, including falsifying examples—not a new generic root-cause list.
