# CODEX — THIRD-PASS CAUSAL DEBUG: TEMPORAL CAMERA REFERENCES, DEPTH CREASES, AND MOVING-POINT REACQUISITION

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Purpose:** Explain the *initiating* erroneous update in the **standard directional forward F1 IEKF** and the first subsequent loss of tracking. Test whether camera-specific frame-0 references eliminate the spurious drive **without destroying real motion evidence**.  
**Mode:** **DEBUG AND OFFLINE COUNTERFACTUALS ONLY. NO PRODUCTION INFERENCE CHANGE.**

> **This document supersedes any implied request to implement a new estimator.** Implement the diagnostic software and tests specified below. Do **not** modify or commit the actual IEKF, noise model, config, solver, RTSS/IEKS, or training pipeline. The main output is auditable numerical evidence that supports or rejects a causal explanation. Do not assume that temporal differencing is a validated production solution.

---

## 0. Evidence already obtained — do not overwrite or misreport it

The complete previous point-level run (`second_pass_20261030.log`) reports 17 static/control/knife-edge point IDs and three additional moving traces. Reuse its output under:

```bash
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
SECOND="$FOR/second_pass_20261030"
```

Read the **actual** `SECOND/SECOND_PASS_ROOT_CAUSE.md`, `SECOND/*.csv`, provenance and image patches before implementing anything. The summary and log are *not substitutes* for the original per-iteration numerical records. If a file is missing, mark it missing; do not reconstruct invented values.

### Corrected previous conclusions

1. **Storage, point 8444, frame 1:** exactly **camera 1** is admitted at the initial predicted state. Its per-channel RGB disagreement is about `(1.6, 1.2, 1.4)` normalized standard deviations. This disagreement at fixed `x_ref` is unchanged through frames 0, 1 and 2. The original accepted displacement is approximately **24.1 mm**, with **1.41 m/s** velocity. The same camera row remains, although its **pixel cell changes**. Original model cost decreases from approximately **7.27 to 2.93**. Depth produces approximately **0.01 mm** of the correction. An attempted frame-0 fusion **did not** eliminate the frame-1 correction; an approximately **31 mm / 1.84 m/s** false update remains under that alternate schedule.
2. **USB, point 67827, frame 1:** **camera 6** has the causally relevant admitted depth crease at approximately **−41 depth sensor σ**. This residual is already present at frame 0; dropping camera 6 eliminates the correction in the previous replay. The same-row objective decreases approximately **1678 → 1.35**. **Camera 4, whose crease was previously reported as −80σ, was pixel-ownership suppressed and DID NOT enter production's update.** Do not attribute the update to camera 4.
3. **Storage, point 8416:** a sub-picopixel perturbation changes the accepted correction substantially; prior replay reports about **12.8 mm versus 6.6 mm** for a `10^-12` px shift, with other cases showing still larger divergence. This is an important *independent* numerical issue; do not confuse it with the origin of a constant image residual.
4. **USB, point 87290:** the last camera rows disappear through the `rgbd_chi2` gate as prediction diverges. By frame 3, the predicted position is about **116 mm** from the GT-propagated position; some cameras still see the corresponding surface according to the prior rendered-depth diagnostic. The exact gate reason must be verified camera by camera.
5. **Erratum:** Storage 262894's previously described cameras 3/4 occluder rows, and Blade 14104's camera-0 occluder row, were **not actually used**. Never present probe-only rows as admitted rows. Row sets must follow production's frozen pixel-ownership and active-point logic.
6. The existing forward F1 algorithm (not IEKS) has `P_vv,0 = I (m/s)^2` and `dt = 1/59 s`. Its position correction can inject `Δv ≈ Δx/dt`; the amplification has already been established. **Reproving that equation is not this task.**

The outstanding questions are: (i) Is the *pre-existing camera-specific residual* sufficient to explain the first wrong movement? (ii) Is the depth crease a demonstrably invalid physical measurement, not just a very large normalized innovation? (iii) Is moving-point loss caused by failure to reacquire a still-visible surface, and precisely which gate rejects it? (iv) Is there *independent temporal evidence* for real moving points under a camera-reference diagnostic?

## 1. Strict scope and immutable invariants

- **Standard F1 only.** No RTS, IEKS, EM, static/dynamic classifier, new state, supernodes, shared rigid constraints, renderer-driven training, new optical flow or optimizer in inference.
- No modifications to `train.py`, `estimator/*.py`, `scene/*.py`, `configs/*.json` or their effective defaults. In particular, do not adjust `P0`, `q_c`, `R`, noise calibration, pixel assignment, gates, interpolation, iteration count, or frame-0 handling in production.
- Use previously saved `USB_100109` and `storage_45135` baseline/traced states and the original RGB-D frames. Blade only if needed for the existing 10251 depth-gradient control; no additional scenes.
- **Do not execute any full-scene standard F1 run, temporal run, Laptop run, full-scene benchmark or batch sweep.** All experiments are **CPU offline single-point/single-frame replays**, with a few frames per point. Reuse frame caches and old traces. Keep the current 4070/5090 GPU free.
- Synthetic ground truth (part transforms, geometry or ray casting) is **only for post-hoc diagnosis/oracle evaluation**. Never use GT to choose an admitted inference row, change a covariance, select parameters, optimize an inference state or declare a non-GT implementation valid. Do not conflate `STATIC_PART` with physical material-point identity.
- Read-only counterfactuals must save into a **fresh directory**, e.g. `$FOR/third_pass_temporal_reference_20261030/`; never overwrite existing traces, configs, baseline outputs, first- or second-pass reports.
- Maintain and log exact original point IDs, original camera IDs, frame IDs, selected-point source camera and original sampling coordinates. One audit row must explicitly say whether the camera was admitted by F1, suppressed by ownership, invalid due to the mask, or excluded by a different gate. **Potentially sampleable != used.**
- If reference evidence is unavailable or invalid, mark `NO_VALID_REFERENCE`; do **not** silently set the reference mismatch to zero or substitute a nearby/other-camera pixel.
- Keep findings and test failures visible. No renaming a failing assertion into a relaxed acceptance test; no fabricated trace precision or unverified success.

## 2. Source-code entry points to inspect, not modify

At minimum examine these actual files and identify their *current local version* and source hash:

- `tools/f1_second_pass.py` — reuse or extend its read-only per-point replay and 1D objective/path machinery;
- `tools/root_cause_oracle.py`, `tools/root_cause_trace.py`, `tools/forensic_f1_drift.py`, `tools/audit_static_equilibrium.py` — existing oracles, trace reading, GT-only selection, provenance and covariance;
- `estimator/free_projective_run.py::run_free` — frame-0 source backprojection, no frame-0 multiview update, full-scene F1 outputs;
- `estimator/point_kalman_cuda.py::PointMeasurement`, `update_points` — exact frozen ownership, iterated MAP normal system, chi-square acceptance and fallback;
- `estimator/projective_terms.py::build_terms` — `r = [a - I; Z - D]`, `H`, `R`, prior-linearized innovation, `S`, NIS and gate;
- `estimator/projective_sensor.py::sample` — metric/inverse depth modes, exact derivatives and variance propagation;
- `estimator/projective_visibility.py` — per-frame pixel ownership, self-occlusion, final recheck;
- `scene/endpoint.py` and camera calibration/data loaders — source observations, world-to-camera projection, depth convention and synthetic frame timing.

The local source may contain uncommitted changes not present on GitHub. **Audit local files before borrowing any file paths or function signatures from this specification.** No `git checkout`, automatic formatting of production files, branch switching or commits.

## 3. The EXACT diagnostic to test: persistent spatial bias vs temporal change

Let original immutable endpoint world position be `x_ref,i`, source-camera RGB be `a_i`, camera `c` be a fixed calibrated camera, and `π_c(x)` and `Z_c(x)` be the production projection and camera-axis metric depth. Define the **original** 4D measurement residual using actual measured RGB-D:

\[
\mathbf r_{c,t}(\mathbf x)
=\begin{bmatrix}
\mathbf a_i-\mathbf I_{c,t}(\pi_c(\mathbf x))\\
Z_c(\mathbf x)-D_{c,t}(\pi_c(\mathbf x))
\end{bmatrix}.
\]

At the **fixed original position**, compute the pre-existing camera-specific discrepancy:

\[
\mathbf b_{c,0}=\mathbf r_{c,0}(\mathbf x_{\rm ref,i}).
\]

For a candidate position at time `t`, define the **reference-differenced diagnostic residual**

\[
\boxed{
\mathbf r^{\Delta}_{c,t}(\mathbf x_t)
=\mathbf r_{c,t}(\mathbf x_t)-\mathbf b_{c,0}.
}
\]

**Exactly** (with the same static camera calibration):

\[
\begin{aligned}
\mathbf r^{\Delta,RGB}_{c,t}(\mathbf x_t)
&=\mathbf I_{c,0}(\pi_c(\mathbf x_{\rm ref,i}))-\mathbf I_{c,t}(\pi_c(\mathbf x_t)),\\
 r^{\Delta,D}_{c,t}(\mathbf x_t)
&=[Z_c(\mathbf x_t)-D_{c,t}(\pi_c(\mathbf x_t))]\\
&\qquad-[Z_c(\mathbf x_{\rm ref,i})-D_{c,0}(\pi_c(\mathbf x_{\rm ref,i}))].
\end{aligned}
\]

At `x_t = x_ref,i` in unchanged images, `rΔ = 0` algebraically. This **is expected** and **is not, by itself, evidence of a usable dynamic estimator**. Demonstrate whether the original false correction was driven by `b_c,0`, and separately demonstrate whether the differenced observable has sensitivity to real moving trajectories.

### 3.1 Non-negotiable mathematical boundaries

1. `b_c,0` is computed once at **immutable `x_ref`** from frame-0 raw images. It is **not** recomputed at the current iterate, accepted position, or true moving position. Recomputing the reference as the point moves changes the model and destroys the experiment.
2. `rΔ` is a *counterfactual measurement residual*, not a proof of a correct generative likelihood. The appearance in camera `c` at frame 0 may belong to a different surface, or its illumination/BRDF may change. A frame-0 reference must pass independent validity/visibility inspection; otherwise report invalid reference.
3. The source-camera reference color and source depth have already defined `a_i`, `x_ref`, and `P_x0`. They are **statistically correlated with the prior**; never treat them as a new independent, double-counted frame-0 measurement.
4. For fixed `x_ref` and fixed reference, the derivative of `rΔ` with respect to the current `x_t` is exactly the derivative of the original `r_t`. The reference offset changes **the residual**, not the Jacobian. Verify this analytically and by independent finite differences **away from pixel boundaries**. Remember the code stores the Jacobian of `h=-r`, so check the sign convention explicitly.
5. A row with no valid frame-0 camera-specific observation must not be assigned a bias of zero. Mark it ineligible for this diagnostic. In moving-part tests, occlusion at either time is a genuine limitation; do not interpolate across masks or depth discontinuities to manufacture the reference.
6. Do not derive a temporal-difference likelihood by assuming all references and current samples are independent when they are not. See Section 4.
7. **Never use `rΔ` to replace the production F1 residual in this task.** The counterfactual exists in an offline replay function only.

### 3.2 Required source and camera consistency checks

For each investigated `(point, camera, frame)` report:

- point ID, source `(frame, camera, pixel)`, 3D `x_ref`, source RGB `a`, fixed original `π_c(x_ref)`, camera depth `Z_c`;
- source frame-0 measured pixel, mask, raw validity; frame-0 reference sample at the fixed projected location; frame-t sample at the **same fixed 3D position**;
- original 4-vector `r0` and `rt`; difference `rt-r0` and whether `rt` and `r0` are identical within exact image-storage precision, *not just within noise σ*;
- camera-wise `R0`, `Rt`, `H0`, `Ht`, raw sample weights/pixel cell and observed depth discontinuities; normalized residuals may be shown with clearly stated variance assumptions;
- for every camera: `USED_IN_F1`, `FROZEN_DUPLICATE`, `SELF_OCCLUDED`, `MASK_EXCLUDED`, `DEPTH_DISCONTINUITY`, `GATE_REJECTED`, etc. Preserve multiple simultaneous reasons where possible;
- tests for camera pose/intrinsics/time being unchanged and frame indices correctly matched. If images are not actually stationary (moving occluder, lighting), report that fact.

Do **not** treat a camera's image as a valid view of the point simply because its projected `2x2` cell is inside the hard mask.

## 4. Correct uncertainty bookkeeping for the temporal diagnostic

For a *scalar or vector measurement channel* with reference noise `n0` and current noise `nt`, the differenced noise is `nt-n0`, so

\[
\boxed{R_{\Delta,t}=R_t+R_0-C_{t0}-C_{0t}},\qquad C_{t0}=\operatorname{Cov}(n_t,n_0).
\]

For two different times `t,s`, reusing the **same frame-0 reference** also creates cross-time covariance:

\[
\boxed{\operatorname{Cov}(n_t-n_0,n_s-n_0)
=C_{ts}-C_{t0}-C_{0s}+R_0.}
\]

Implement **three clearly labeled variance/uncertainty audit modes** in offline analysis only:

- `ALGEBRA_ONLY`: hold `R_t` identical to baseline and change **only** `r_t → rΔ`. This isolates the deterministic residual-bias effect but **does not** claim a statistically valid differential likelihood. This is mandatory for a clean causal comparison.
- `INDEPENDENT_SAMPLES_APPROX`: use `R_t+R_0` for a single-frame update, if and only if both samples are modeled as independent; clearly state this is an assumption and **is not valid** when the reference shares noisy source-pixel information with `x_ref/P_x0`, when physical image noise is shared between frames, or where spatial footprints overlap. It must not be applied as an unqualified production covariance.
- `MEASURED_CORRELATION_DIAGNOSTIC`: only when empirical pixel/reference covariance is actually available, calculate `C_t0`, `C_0t`, possibly cross-channel and cross-camera blocks and use the formula above. Otherwise emit `NOT_ESTIMABLE_FROM_AVAILABLE_DATA`; do not invent a correlation coefficient or add an arbitrary variance floor.

For changing candidate positions, interpolation weights and sensor-noise covariance can be position-dependent; document whether the replay uses candidate-specific `Rt` or the original fixed-reference linearization. For the original source-camera measurement, warn about correlation between canonical color, geometry and reference; report independence-based numbers **only as sensitivity analyses**.

**Do not sum framewise reference-differenced residuals as independent evidence over multiple frames**; the shared reference introduces off-diagonal temporal covariance. The requested experiments use **single-frame** replays precisely to avoid silently building an invalid temporal estimator.

## 5. Mandatory experiment A — Storage 8444: demonstrate and dissect the false RGB motion

Use original **Storage point 8444**, decisive **frame 1**, **admitted camera 1** only. Include fixed-point frames 0,1,2 and stable control(s) such as Storage 60019/100564 if actually present in the saved second-pass selections.

### 5.1 Reproduce production BEFORE analyzing alternatives

- Load the exact saved `s_minus`, full `P_minus`, noise policy, RGB/depth source, original predicted pixel-ownership map and frame-1 raw image.
- Reproduce all accepted-camera IDs, every IEKF iteration `j`, original RGB/depth residuals, `H`, `R`, `A`, `b`, step, 2×2 pixel cell, total cost, chi-square statistic, accepted/fallback state and final `Δx`, `Δv`.
- Verify original accepted position and velocity with the independent oracle to a numerically justified tolerance. For nonconverged/knife-edge examples, do not silently relax parity; log precise rounding divergence by iteration.
- Demonstrate whether **camera 1's reference row is geometrically and photometrically valid at frame 0**. The fact that it produces an RGB value does not prove material identity.

### 5.2 Controlled replay arms (all start at the identical `s_minus/P_minus`)

Run the following *offline*, freezing whole-scene occupancy and preserving original rows where physically evaluable. Reuse the **same** IEKF inner-iteration solver; no learning/tuning. Score the new residual model with its own correctly named objective; **never directly compare numeric costs across different residual definitions as if they were the same likelihood**.

| Arm | Change in offline replay | Question |
|---|---|---|
| A0 | Baseline original residual, exact original `R_t` | Reproduce `Δx ≈ 24.1 mm`, `Δv ≈ 1.41 m/s`, `J_orig: 7.27→2.93` |
| A1 | RGB only: `r_RGB,t → r_RGB,t - r_RGB,0`; original depth residual and original `R` | Does the static RGB push vanish without removing depth information? |
| A2 | Depth only: `r_D,t → r_D,t - r_D,0`; original RGB residual and original `R` | Negative control: should not cure a truly RGB-driven case |
| A3 | Both RGB and depth differenced, `ALGEBRA_ONLY` | What remains when all persistent camera residuals cancel? |
| A4 | Same A1–A3 but with independent-reference variance approximation if justified | Does an honest variance increase materially change the local result? |
| A5 | Original residual with camera 1 withheld, preserving other frozen admissions | Does the original false motion disappear if the causally decisive camera is removed? |
| A6 | Frame-0 non-source fusion then frame 1 (existing second-pass result) | Reproduce **negative result**; do not claim frame-0 fusion solves this |
| A7 | Fixed `x_ref`, compare original `r_t` with `rΔ_t` for frames 0–5 | Is the spatial discrepancy persistent with zero temporal difference? |

For every arm record at least: admitted camera set at initial prediction and final candidate, per-iteration `Δx` and `Δv`, original/current 2×2 cells, `r0`, `rt`, `rΔ`, same-model `J_before` and `J_after` on explicitly stated support, algorithm acceptance, and rejected/invalid rows.

**Essential warning:** If A1 cancels camera-1 residual exactly at `x_ref` and causes `Δx=0`, that is an algebraic consequence of the chosen reference, **not sufficient evidence that we can track moving objects with this measurement model**. This experiment answers only the *origin of this static point's false update*.

### 5.3 A physically meaningful cross-check

Compare original camera 1's frame-0 and frame-1 RGB image values at the same projected physical location, including the actual 2×2 neighboring quantized pixels and the stored RGB color encoding. Check whether the discrepancy arises from constant cross-view appearance/quantization versus depth/geometry misprojection. Identify which hypothesis is still indistinguishable with these data; do not infer a BRDF model without evidence.

## 6. Mandatory experiment B — USB 67827: disentangle the persistent depth-crease misfit

Use original **USB point 67827**, decisive **frame 1**, **admitted camera 6** only for the identified crease mechanism (other admitted cameras can contribute and must remain in baseline replay). Distinguish the other admitted **camera 5** explicitly. **Do not use suppressed camera 4 as a causal factor.**

### 6.1 Local geometry and reference audit

- Display (numerical tables plus PNGs) raw depth and foreground-mask 5×5 or 7×7 patches centered around the projected location in **camera 6** at frame 0 and frame 1; keep identical pixel coordinates and a consistent color scale.
- Record the four 2×2 depth corners, their arithmetic and inverse-depth interpolation, first differences, horizontal/vertical second differences in the wider neighborhood, bilinear gradient, local estimated normal, and the original `same_surface` threshold and row-gate statistic.
- Show explicitly whether the crease is temporally **unchanged**. A constant geometric mismatch alone supplies no evidence of actual temporal motion.
- Fit one local inverse-depth plane to each geometrically coherent side of the crease **if** the patches permit an unambiguous side partition. Do not fit a plane across the crease and call it the correct surface. Save fit residuals and confidence/validity; use a held-out pixel subset for the geometric consistency check when possible.
- If the Sapien mesh/ray intersection for camera 6 can be obtained **without changing inference**, compare camera-ray depth for the original physical surface/face and observed rendered depth. Mark **GT_ORACLE_ONLY**. If mesh/material identity cannot be obtained, label `FACE_IDENTITY_UNPROVEN`: a sparse part label does not prove the original face.
- Explicitly assess whether camera 6's four neighboring pixels can plausibly arise from a *single* local planar patch. Report `CONSISTENT`, `INCONSISTENT`, or `UNDETERMINED`, with the evidence and uncertainty assumptions; no ad-hoc segmentation threshold converted into a production policy.

### 6.2 Controlled replay arms

| Arm | Offline intervention | What it tests |
|---|---|---|
| B0 | Original baseline exactly as production | Verify accepted `Δx`, `Δv`, current-row objective and occupancy |
| B1 | `r_D,t → r_D,t-r_D,0` **camera 6 only**, baseline RGB and `R` | Does canceling a constant crease residual eliminate its false force? |
| B2 | Camera-6 depth reference only; all eligible cameras' RGB differenced separately | Separate depth-induced force from appearance effects |
| B3 | Fully differenced RGB and depth for valid cameras, `ALGEBRA_ONLY` | Test combined persistent-offset hypothesis |
| B4 | Remove camera 6 and replay; keep other actually admitted rows | Reproduce previous necessary-camera intervention |
| B5 | Replace *only camera 6's sampled depth* by a physically validated same-surface local model, if one can be established **without GT-dependent inference** | Diagnose whether interpolation across the crease is necessary. Otherwise `NOT_EXECUTED: UNPROVEN_SURFACE` |
| B6 | Original baseline, but fixed initial camera-row membership and log all cell switches | Did the correction change to a different face within an admitted camera? |
| B7 | Probe suppressed camera 4 **diagnostic only**; ensure zero contribution to the production information sum | Regression against the previous incorrect attribution |

For B5, the replacement is a **post-hoc numerical oracle**, not a proposal to use a mesh-based measurement or oracle correction in the actual filter. Even if the replacement removes drift, evidence for the *real* camera-6 surface identity still needs to be stated.

Required verdict: distinguish (a) persistent depth residual that is falsely treated as movement, (b) structurally mixed-surface depth, (c) numerical cell-crossing/row-support effects and (d) unchanged baseline noise/prior amplification. These may all contribute; report their separate and combined effect.

## 7. Mandatory experiment C — USB moving point 87290: the EXACT first failed reacquisition

Use original **USB point 87290** and optionally **85696** as a second failure mode. Reconstruct the first frame at which all cameras have **zero accepted rows**, the last successful preceding frame, and frame 3 where the reported prediction error is roughly 116 mm. Use the saved full-scene state for frozen pixel occupancy, not an isolated point cloud whose visibility differs from production.

### 7.1 Camera-by-camera two-location diagnostic

For every camera, at both:

1. **`x_pred`:** the original standard F1 predicted position;
2. **`x_GTprop`:** GT part-transform propagation of the original frame-0 point, **only as a diagnostic** (not assumed exact material-point registration);

report:

- projected `(u,v,Z)`, 2×2 cell/weights, raw foreground mask, measured depth and color, measurement availability, and all distinct rejection causes;
- frozen occupancy winner/contenders, whether candidate row was suppressed by pixel reuse, predicted self-occlusion or final recheck; do not equate sampleability with an admitted row;
- original raw depth difference `Z-D`, RGB difference, row covariance `R`, predicted innovation covariance `S`, *prior-linearized* innovation `nu`, Mahalanobis statistic, threshold, final gate code;
- distinguish (i) the **measurement residual at the candidate location**, (ii) the gate's **innovation linearized back to the original prediction**, and (iii) hypothetical innovation **if that candidate were the prediction**; the latter is **GT_ORACLE_ONLY**, not an inference update;
- synthetic rendered depth or ray-cast visibility check if physically meaningful; if only the object's part is visible but material point visibility is unknown, label `PART_SURFACE_VISIBLE_MATERIAL_POINT_UNKNOWN`;
- predicted and GT-propagated position errors, current/previous velocity, support history, and first nonzero innovation that led to divergence.

### 7.2 Reacquisition causality table

For the *last originally usable camera* record side by side:

| State/counterfactual | Mask | Occlusion | Pixel owner | `r_D` | `R_D` | `S_DD` | `NIS` | Gate | Intended interpretation |
|---|---|---|---|---|---|---|---|---|---|
| Original `x_pred` with original prior and occupancy | | | | | | | | | Actual F1 rejection |
| `x_GTprop` with original prior and occupancy | | | | | | | | | Can the true vicinity produce a useful sample? |
| `x_GTprop` with prior hypothetically recentered for oracle | | | | | | | | | Is rejection caused by the wrong predicted mean? |
| Original `x_pred` with RGB/depth temporal reference *if valid* | | | | | | | | | Diagnostic only, no claim of real reacquisition |
| GT-propagated point with actual full-scene owner map | | | | | | | | | Does competing occupancy block otherwise valid evidence? |

Do **not** take a formerly rejected pixel and call it a valid correspondence merely because it lies inside the mask. If the correct physical point cannot be localized to the pixel, say so.

### 7.3 Check whether temporal differencing destroys moving evidence

At frame 1, 2, and the first loss frame (where a valid camera-specific frame-0 reference exists):

- evaluate original and differenced RGB/depth residuals at (a) original static `x_ref`, (b) production `x_pred`, and (c) diagnostic GT-propagated `x_GTprop`;
- quantify original and differenced 3D Jacobian ranks/eigenvalues, current-row validity, and local residual contrast along a true-motion direction (obtained from GT **only as a sensitivity oracle**);
- test whether a stationary position and the GT moving position are distinguishable under the differenced residual. **A diagnostic that cancels all residuals at the wrong location and also loses response to actual movement is NOT a successful fix.**
- do not mistake `rΔ(x_ref) ≠ 0` when the moving point leaves/gets occluded by a different surface for evidence of correct point correspondence.

Output explicit `MOTION_EVIDENCE_RETAINED`, `MOTION_EVIDENCE_LOST`, or `UNDETERMINED` with measured numbers and exact assumptions. A decrease in a static-point false step is insufficient by itself.

## 8. Controls for model bias, optimizer behavior, and reference selection

Include at least:

- **USB static control 51974** and its drifting neighbor **51975**, with exactly the actual admitted camera rows; compare camera-specific `b_c,0`, and whether the reference changes the first accepted update. A control with no valid new camera rows does not prove that a residual model is stable.
- **Storage stable control 60019 or 100564**, if the existing traces have an eligible frame-0 camera reference; match static/reference visibility, not merely initial point depth.
- **Storage 8416** one-sided pixel-boundary test under original and referenced residuals, with `u/v ± 10^-12`, `±10^-10`, `±10^-8` pixels. Log precise iterations, original/new cell, actual objective, and whether perturbations still cause divergent accepted steps. The temporal reference experiment is **not permitted to mask the existing numerical instability**.
- **USB 30502**, optional but valuable: first accepted row at frame 3, where drift-to-far-face occurs; test whether a frame-0 camera-1 reference exists before attributing the motion to a persistent bias. Do not fabricate frame-0 pixel association if the point was already hidden.

Report the difference between three causes instead of merging them:

1. `STATIC_CAMERA_BIAS`: a persistent camera-specific discrepancy already existed at frame 0;
2. `INVALID_SURFACE_SAMPLE`: a candidate interpolant spans or lands on the wrong visible surface;
3. `NUMERICAL_SUPPORT_ARTIFACT`: the iteration improves its result by changing pixel cells, dropping rows, or depends catastrophically on a nondifferentiable boundary.

A changed pixel cell **is not sufficient** to establish cause (3). It is only evidence to inspect. It may reflect legitimate tracking and a genuine reduction of the same nonlinear residual.

## 9. Objective and acceptance accounting — do not accidentally prove a false monotonicity claim

For every replayed arm:

- Reproduce the original accepted state and production's exact camera-row membership at the predicted state before intervening.
- **Within each specified residual model**, compute `J_prior(s) + Σ rᵀR⁻¹r` (same normalization as the actual solver), both at the predicted candidate and the accepted one. Record row membership and actual pixel cells at both states.
- Separately compute the cost on the **same identifiable set of rows** when both endpoints' samples are geometrically valid; mark rows invalid at either endpoint instead of silently treating them as zero cost.
- Track per-camera and per-channel cost, gradient `g = Σ HᵀR⁻¹r`, position information `F=Σ HᵀR⁻¹H`, eigenvalue spectrum and the accepted displacement's projections onto weak information directions.
- Report original-model cost and referenced-model cost **in separate columns**. Different likelihoods have different normalizers and supports: `J_original` versus `J_reference` values **cannot** be ranked against each other as Bayesian model evidence.
- An accepted iterated step that decreases the **referenced** residual model's cost is not proof that it improves the original physical trajectory. Evaluate with GT only in post-hoc error metrics.
- Do not redefine chi-square gate acceptance silently in the counterfactual. If a row becomes invalid, record why and separate changes in **the residual force** from **support-selection changes**.
- Report number of iterations, convergence, whether fallback was selected, and numerical stability. The baseline code may accept nonconverged states; identify this explicitly.

### 9.1 Diagnostic intervention hierarchy — isolate one mechanism per arm

Each point must have:

`BASELINE`, `REFERENCE_RGB_ONLY`, `REFERENCE_DEPTH_ONLY`, `REFERENCE_RGBD`, `CAMERA_DROP`, `ORIGINAL_FIXED_ELIGIBILITY`, `REFERENCE_FIXED_ELIGIBILITY`; plus optional `INDEPENDENT_SAMPLES_APPROX` variants and geometric-depth oracle arm where valid.

Explain effects in the order:

**raw residual change → gradient change → proposed position step → accepted position step → velocity correction → new predicted location → row admission**, with actual numerical values. Do not jump from a residual reduction straight to a claim of recovered tracking.

## 10. Independent mathematical and computational tests

Create `tests/test_f1_temporal_reference_diagnostics.py` (CPU only), with at least these tests. Run them before interpreting empirical results.

**T0: exact cancellation / sign convention.** Construct an unchanged static synthetic image with a fixed per-camera RGB or depth offset and a stationary projected 3D point. Verify `r_orig = b_c,0` and `rΔ = 0` to floating-point tolerance; verify the analytic derivative of `rΔ` equals the derivative of `r_orig` **away from cell boundaries**. Test both signed residual conventions against the production `h` Jacobian.

**T1: genuine moving signal is not automatically erased.** Use one camera, a textured moving surface with **known correspondences** and physically correct rendering/measurement. Evaluate `rΔ(x_true,t)` vs `rΔ(x_ref,t)` and local sensitivity to movement. Include a textureless plane where tangential motion is structurally unobservable. The test must **not** require all moving cases to be identifiable; instead it must report/verify what is and is not observable.

**T2: correct differential covariance.** Independently sample a small Gaussian joint noise vector with known `R0,Rt,Ct0` (including a correlated case), and verify `RΔ=Rt+R0-Ct0-C0t`. For three frames, verify the shared-reference off-diagonal `Cov(n_t-n0,n_s-n0)`. Compare to dense matrix algebra, not only Monte Carlo. Verify PSD within numerical tolerance and never invert a singular covariance without proper treatment.

**T3: no double counting of source observation.** Frame-0 source depth/color used to define `x_ref/a/P0` must not be entered again as an independent frame-0 measurement. Assert provenance and behavior; if correlation with `x_ref` is unmodeled, the audit must flag it.

**T4: invalid-reference behavior.** When frame-0 reference is masked, occluded, discontinuous or has invalid raw depth, the referenced channel cannot be used; report `NO_VALID_REFERENCE`, and **do not silently substitute** another camera/pixel or a zero residual.

**T5: baseline reproduction.** Replay original Storage 8444 and USB 67827 using the exact old model: state/velocity, accepted camera set and per-iteration record must match preexisting traces to a justified precision; no result from a modified model may be presented as baseline.

**T6: frozen ownership.** Probe-only cameras excluded by frozen pixel assignment (USB 67827 camera 4; Storage 8444 cameras 0 and 7; Storage 262894 cameras 3/4) must contribute exactly zero to the baseline or new replay update unless an explicit, separately labeled ownership counterfactual is being run.

**T7: pixel-boundary sensitivity.** Test one-sided derivatives at integer-coordinate crossings, log divergent IEKF iteration histories and ensure no silent convergence claims. If counterfactual replay isn't bitwise reproducible due rounding, report exact divergence step.

**T8: gate-of-last-camera oracle.** A moving synthetic point with erroneous mean, correct physical surface visible but χ² gate rejection: report baseline gate and the alternate GT-centered oracle gate separately. Prove diagnostic location checks do not alter the real state.

All tests must be deterministic, run without a GPU and make no writes to inference paths.

## 11. Deliverables and file organization

Create **investigation-only** code, preferably extending the existing second-pass tool without duplicating its scene loader and full camera probe logic:

```text
$FOR/third_pass_temporal_reference_20261030/
  README.md
  THIRD_PASS_CAUSAL_RESULT.md
  cases/
    storage_8444.md
    usb_67827.md
    usb_87290.md
    storage_8416.md
    controls.md
  csv/
    reference_residuals_frames0_to5.csv
    differential_uncertainty.csv
    arm_outcomes.csv
    per_iteration_objective_and_rows.csv
    per_camera_depth_crease.csv
    last_camera_reacquisition.csv
    moving_sensitivity.csv
  traces/
    exact_replays.npz
    input_hashes.json
  figures/
    storage_8444_reference_vs_original.png
    usb_67827_depth_crease_frame0_frame1.png
    usb_67827_original_vs_reference_objective.png
    usb_87290_last_camera_gate.png
    storage_8416_one_sided_sensitivity.png
  test_results.txt
```

Data files may be compressed and chunked; **do not** save full-scene arrays, repeated source RGB-D frames, every image patch in an entire sequence or multi-gigabyte dumps. Use selected point IDs and the original published arrays. Keep all figures independently interpretable with units, camera/frame/point labels and the actual support decisions.

Include a new script such as `tools/f1_temporal_reference_diagnostics.py` with `--forensics-root`, `--second-pass-root`, `--out` and optional `--case` selection. It must be deterministic, CPU-only, and must refuse to overwrite a nonempty output directory. Add a `--dry-run` path listing the exact input files/point IDs/frames; verify these before starting. Reuse only local current code and old trace files; never assume GitHub's version contains uncommitted forensic tools.

**Minimal execution plan** (inspect the actual CLI before running; do not invent a working flag):

1. Read the second-pass report and corrected erratum; validate paths/provenance.
2. Implement and run independent unit tests.
3. Replay Storage 8444 and USB 67827 baseline first; STOP and report if these do not match.
4. Run their controlled single-point diagnostic arms and produce primary figures/tables.
5. Run USB 87290 gate/reacquisition diagnostics and Storage 8416 numerical sensitivity control.
6. Run remaining controls only if the primary explanations are numerically supported and the existing files contain their camera/frame data.
7. Write the full result and **a short summary in the terminal**. Do not launch a full scene, Laptop, IEKS or further experiments.

No automatic chain from test completion to any full run.

## 12. Required closing verdicts — no hand-waving

`THIRD_PASS_CAUSAL_RESULT.md` must answer these *separately*:

### V1 — Storage 8444

- Did the causally relevant camera-1 RGB mismatch preexist at frame 0? Give raw values, `r0`, `r1`, `r1-r0` and validity.
- On **identical initial 6×6 prior and frozen candidate occupancy**, did replacing only the persistent camera-specific RGB offset remove the false **position and velocity correction**? Give both exact signed `Δx` and `Δv` and the original/reference within-model cost trajectories.
- Was cancellation purely algebraic at the reference position? If so, state it explicitly. Does this establish a viable motion-tracking model? **No: test moving controls first.**
- What explains the remaining error (if any): depth, support change, numerical divergence, or independent temporal appearance change?

### V2 — USB 67827

- Camera 6 (not camera 4): was its depth reference valid and temporally constant? Give actual 2×2 corners, 5×5/7×7 profile, crease measure, residual, `R`, `S`, NIS, gate, and per-camera information contribution.
- Does subtraction of the constant reference mismatch remove the correction? Does a physically justified same-surface depth oracle support this causal conclusion? **Never infer actual face identity from a static-part label.**
- Can the point still move to the wrong face due to the interpolation/association model despite a reference cancellation?

### V3 — USB 87290

- Name the **last successful update frame** and **first zero-row frame**; name every camera's actual rejection reason at that boundary.
- Would original measurements at the GT-propagated point be geometrically valid? Do they still fail the prior-centered gate, and would they pass if the predicted mean were hypothetically corrected? Label the latter strictly oracle-only.
- Does the reference diagnostic preserve **some independently verifiable motion evidence** on valid original/surface observations? If not, reject it as a candidate production design.

### V4 — Numerical/control evidence

- Quantify one-sided boundary sensitivity on Storage 8416 under original and reference residual models.
- Compare stable/drifting neighbor controls using only eligible camera references.
- State whether remaining instability can be blamed on poor observations, nonconvergent optimization, wrong-surface association, or an unidentifiable motion direction, and what independent evidence separates them.

### V5 — Generalization and actual recommendation

Answer with one of:

- `PERSISTENT_BIAS_CONFIRMED_AS_CAUSAL_FOR_THE_SELECTED_STATIC_CASES`;
- `PERSISTENT_BIAS_PARTIAL`;
- `PERSISTENT_BIAS_REJECTED`;
- `INSUFFICIENT_VALID_REFERENCE_EVIDENCE`.

Also report an **independent** `MOTION_EVIDENCE_RETAINED / LOST / UNDETERMINED` outcome. Do **not** report overall success unless both static false correction reduction and moving-point evidence preservation are supported by measurements.

State exactly which part of the original physical model is falsified and which remains unresolved. Explicitly separate:

1. **Observation-model cause** (pre-existing camera bias or discontinuity);
2. **Numerical optimizer cause** (nonconvergence, cell-boundary sensitivity or changed row support);
3. **Kalman amplification** (`Δx → Δv`, already understood);
4. **Later loss of physical point tracking** (visibility/gate/ownership).

**The acceptable output is a causal explanation with reproducible single-point counterfactuals. The unacceptable output is a redesigned estimator, tuned priors, a lower static residual without preserved moving evidence, another generic velocity-gain derivation, or an expensive dataset run.**

---

## 13. Suggested concise task message to Codex

> Read and execute this Markdown document as a **debug-only** implementation contract. Do not modify the working IEKF or run any complete scene. Start by reading the second-pass report and confirming the erratum about frozen-out cameras. Reproduce Storage 8444 and USB 67827 exactly, then run controlled, uncertainty-labeled camera-specific frame-0-reference residual counterfactuals. Independently audit USB 67827's crease geometry and USB 87290's last-camera χ² rejection and oracle-reacquisition. Include moving-point evidence-preservation and Storage 8416 pixel-boundary tests. Provide point-by-point numerical causal verdicts, not a proposed new estimator; stop if the baseline replay cannot be reproduced.
