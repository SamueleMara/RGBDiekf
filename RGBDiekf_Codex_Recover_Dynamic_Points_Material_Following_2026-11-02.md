# RGBDiekf — recover moving points without reintroducing static drift

**Implementation and falsification contract for Codex · 2026-11-02**

## 0. Verified baseline and scope

- Public baseline examined: `SamueleMara/RGBDiekf` commit `9465cb109f22e4047d7132b3f9d2f22f3f5d633e`. The new `local-correlated` implementation and report `reports/temporal_local_correlated_validation.md` are described by the user but **not committed at this verified baseline**: inspect the actual working tree, diff, and the full local report before modifying them. Preserve uncommitted work.
- Codex's reported data: USB and Blade frames 0–2, eight calibrated cameras, full cloud. Static drift essentially removed, but moving point displacement estimates remain nearly zero. USB moving GT displacement 15.7 / 31.3 mm; Blade 14.2 / 28.3 mm. Predicted projection ~1.3 px away each step. USB 18,639 / 18,928 rejected full steps violate frozen-current-pixel/row-domain policy. Average accepted updates ~0.6 mm. At GT pose old warp depth normalized residual ≈8.2 USB and ≈59,000 Blade against expected ≈3.4. These are **reported results**, not independently rerun in this chat.
- 106 focused tests reportedly pass, including local Gaussian numerical parity; that verifies algebra on fixtures, **not moving point recovery**.
- Objective: estimate **individual dynamic point** `s=[x,v]` from RGB-D and masks. Maintain old baseline, simplified, globally correlated full and existing local-correlated modes untouched for A/B comparisons. Do not introduce rigid-part labels, geometric/adjacency graph, 3DGS, normal, surfel, mesh, GT-informed priors, velocity clamps, arbitrary gates, or fabricated warp noise.
- Current code-level reference: `estimator/temporal_rgbd_terms.py:temporal_prediction`, `build_temporal_rows`, `validate_reference_patch`; `estimator/temporal_zbuffer.py:first_hit_pixel_owner`, `selected_owners`; `estimator/temporal_simplified_iekf.py:update_points_temporal_simplified`, `point_domains`; `estimator/temporal_sparse_nonlinear.py:evaluate_nonlinear_objective_and_visibility_gpu`, `solve_nonlinear_sparse_iekf_gpu`; `estimator/temporal_sparse_run.py:same_current_domain`; `estimator/temporal_reference.py:ReferenceTable`; `estimator/point_kalman_cuda.py:transition,predict`; `train.py` and `tools/run_temporal_rgbd_critical.py`. The uncommitted local solver has additional files: discover them, don't assume filenames.

## 1. The precise reason zero motion survives every covariance backend

The current Eulerian residual attaches to a CURRENT INTEGER pixel `p`, then looks backward to `q=p-[pi(x_t)-pi(x_ref)]` and compares `I_t(p)` vs `I_ref(q)` and `D_t(p)` vs `D_ref(q)+deltaZ`. In addition, the acceptance policy often requires the selected `p`/owner/reference row signature to remain constant. When `pi(x_t)` moves >0.5 px, first-hit ownership and/or `p` must change. Thus genuine motion is rejected as a domain change. A 6x6 prior, a local correlated posterior, or a globally exact posterior cannot repair this observation-association contradiction.

**Do not merely disable the row-signature check:** the old residual at a frozen current pixel may be comparing a different surface. A step accepted with a disappearing row is also not evidence of improved fit.

The second separate issue is the GT-true residual: the old differential depth warp compares Eulerian pixels with potentially different surface identities. The very small declared Blade depth variance amplifies model discrepancy into enormous normalized residuals. A GT-residual failure cannot be cured by covariance algebra; measure support/ownership/resampling bias *before* selecting a warp-model discrepancy covariance.

## 2. Replace the observation by a material-following, source-anchored RGB-D residual (NEW OPT-IN MODE)

For tracked point `i`, camera `c`, and its most recent valid accepted SAME-CAMERA reference frame `tau`, store the associated **measured source pixel** `p_ref` and its original sample, plus reference state `s_ref` and the local joint covariance. At candidate state `s_t`, let

```
u_t   = pi_c(x_t)
u_ref = pi_c(x_ref)
Delta_u = u_t - u_ref
q_t = p_ref + Delta_u      # continuous/subpixel coordinate in the CURRENT image
Delta_Z = Z_c(x_t)-Z_c(x_ref)
```

Use the current observation at that *moving* material coordinate; the reference data remains at the fixed material anchor pixel. Define, for one reference RGB-D pixel `p_ref`,

```
r_RGB = bilinear(I_c,t, q_t) - I_c,tau(p_ref)
r_D   = bilinear(D_c,t, q_t) - D_c,tau(p_ref) - Delta_Z
```

This is a **proposed change of temporal observation domain** and must be labeled accordingly (not falsely claimed algebraically equivalent to the old fixed-`p` Eulerian residual). For subpixel `p_ref` use an appropriate source interpolation with its actual raw pixel set and correlations. The original source observation must not be counted twice.

Residual derivatives (position columns; velocity columns ZERO):

```
dr_RGB / dx_t    =  grad I_t(q_t) @ J_pi,t
 dr_RGB / dx_ref = -grad I_t(q_t) @ J_pi,ref
 dr_D   / dx_t   =  grad D_t(q_t) @ J_pi,t - J_Z,t
 dr_D   / dx_ref = -grad D_t(q_t) @ J_pi,ref + J_Z,ref
```

If `p_ref` is itself a state-dependent interpolated coordinate, include its full derivatives as an alternative **only after deriving and testing**; do not silently mix anchor conventions. Derivatives above are for fixed measured `p_ref` and the defined `Delta_u`.

Let `a=1..4` be current bilinear contributors at `q_t`. The raw current sensor covariance is a 4x4 RGB-D covariance per raw pixel; its interpolated contribution is `sum_a w_a^2 R_t,a`. The reference contribution is `R_ref` if reference uses one raw pixel; if it uses bilinear interpolation use actual weights and pixel IDs. Use the implemented **within-point joint Gaussian** to retain cross-time state/reference-pixel correlations, rather than adding marginal terms twice. `R_warp` is additional model discrepancy only; `ideal` explicitly means **zero extra discrepancy in a falsification diagnostic**, not zero raw sensor uncertainty.

The new model is a material point observation, not a rigid patch. A single pixel's RGB/depth is sometimes insufficient to disambiguate tangent motion; report that objectively using the local Jacobian information rank/eigenvalues, no fabricated motion.

### Visibility and mask rules

1. Use the full set of all tracked points and calibrated cameras for projection/depth sorting. The selected point may change CURRENT pixel when moving. It must never be forced to retain its old `p`.
2. Validate **CURRENT** image bilinear support at `q_t` (foreground mask for every contributing raw pixel, valid finite depth, single-surface consistency using existing explicitly documented policy). Do not use background RGB or depth.
3. Check first-hit occlusion at the candidate state. A changed winning point or pixel is an association event to validate, **not automatically an error**. When hidden, use prediction-only for that camera; do not force a background update.
4. The reference pixel/frame/camera must still refer to the same point's accepted same-camera observation. On initial view in that camera, bootstrap without temporal row. On genuine occlusion, preserve reference if still in the lag window. Do not cross cameras for RGB identity.
5. Reject decreases caused solely by deleting invalid rows. Compare objectives on a stable set of point-camera **material identities**, and separately report reactivated/lost observations; recompute owners after candidate update. A fixed material identity is NOT a fixed image pixel. Avoid turning an occlusion change into an invented 3D force.

## 3. First run a decisive GT-only physical falsification analysis (NO inference access to GT)

Implement `tools/audit_temporal_dynamic_correspondence.py`, separate from estimation. For the real SAPIEN USB and Blade frames 0/1/2 and independently identified moving/static points (using GT *only here*):

- Confirm GT point identity and world coordinate correspondence in both frames, optical camera conventions, timestamps and source cloud registration; if not grounded, report `GT_CORRESPONDENCE_UNVERIFIED`.
- For each camera compute true `pi(x_ref_GT)`, `pi(x_t_GT)`, `Delta_u_GT`, `Delta_Z_GT` and visible/frontmost status using complete candidate geometry. Distinguish camera-visible true material points, target pixels occupied by another surface, foreground edge, depth discontinuity and textureless patch.
- At these exact true displacements measure: (A) old fixed-current-pixel warp residual, (B) new moving-current-image residual at `q_t`, (C) direct geometric consistency `D_t(pi(x_t_GT)) - Z_t_GT`, (D) `I_t(pi(x_t_GT))-I_ref(pi(x_ref_GT))` with the same source-pixel convention. Compare raw residual RGB 0–1, depth metres and normalized residual against the actually declared noise. Report zero/additional noise premise, calibration provenance and interpolation bias. Do **not** use a guessed variance to make statistics look good.
- Report per scene/camera/frame/point-class: N eligible, N occluded, N four-neighbor-invalid, N large residual at true motion, median/p90/p99 absolute residuals, normalized residual, Hessian spectrum/observable motion subspace, source-current pixel shift. Plot residual vs distance to depth edges and vs surface orientation proxy if available from depth gradients (do NOT introduce normals into the estimator).
- Save 20+ representative moving-point traces, including USB 67827/87290/85696 and Blade 27016 **only if they are present and have verified GT source correspondence**; also trace static references. Include camera pixel crops and GT projected centers. GT information must be post hoc, never used to select proposals, noise or parameters during filter inference.
- **Gate A:** If the new residual at GT is still strongly biased after restricting to genuinely visible same-surface observations, stop. Diagnose a physically invalid pixel-transport model (e.g. sampling, sensor depth convention, color/appearance change, source pixel/geometry mismatch). Do not optimize a demonstrably wrong likelihood.

## 4. Recover association and motion without imposing motion between points

### Stage B1: projection-continuous local IEKF

Add a new opt-in local mode `material-following` using the existing local-correlated 6D point state plus its within-point temporal references. Accept changes of target integer pixel; do not require `point_domains` / `same_current_domain` equality of CURRENT pixel. Keep accepted correspondences tied to material identity `(point_id,camera,reference_frame,source_pixel)` and validate candidate support and visibility dynamically.

Run multiresolution image pyramids (same-camera RGB-D + corresponding hard masks), from coarse to fine, to enlarge the basin of the *same material-following photometric/depth likelihood*. No regularization linking distinct point velocities or point-to-point motion graph. Pyramids must preserve camera scaling and metric-depth conventions, validate gradients on every level, and correctly propagate source and target pixel covariances. Report level-by-level motion, gradient/information rank and accepted observation support. If genuine motion is obtained at original resolution without a pyramid, prove it on both scenes and keep pyramid optional.

### Stage B2: optional explicit image correspondence proposal, only if B1 lacks a motion basin

Construct a **data-only** same-camera source-to-current subpixel correspondence from foreground RGB-D (for example coarse-to-fine local source patch matching; no pretrained flow model, smoothness/rigid priors, GT, or point-neighbor motion coupling). The patch may use neighboring **image measurements**, but must not define a rigid 3D point neighborhood. Use only valid same-surface reference/current foreground pixels. Evaluate multiple candidate displacements if a local minimum would trap the update. The search domain may be chosen from and reported against the predicted projected covariance, with observed ambiguity explicitly returned as uncertainty; do not hard-code a GT displacement, a static/dynamic partition, or an arbitrarily truncated motion limit.

The proposal is an **association initializer**, not a pseudo-ground-truth motion observation. The final IEKF must still evaluate the physically specified moving-sample RGB-D likelihood and its covariance at the proposed state and reevaluate current first-hit ownership. If a separate 2D flow measurement is assimilated, derive its covariance/cross-covariance and show that underlying RGB-D data are not counted again independently (otherwise use it for initialization only).

Without sufficient image evidence a point's tangential motion is unidentifiable; keep it prediction-only with honest uncertainty and report the observability shortfall. **Do not claim all dynamic points must be recovered by arbitrary per-point RGB-D measurements without texture/association information.**

## 5. Covariance and numerical requirements

- Reuse `local-correlated` pointwise Gaussian machinery and original physical `F(dt),Q(dt)`. Keep correct cross-time `Cov(s_t,s_ref)` and same-point shared reference-pixel covariance. Approximation across different points must remain explicit; do not claim global exactness.
- Do not reset posterior covariance when changing reference. New valid camera reference replaces/extends the same point's local random variables with the correctly conditioned covariance; no independent re-sampling of a correlated measurement.
- Do not conflate `R_sensor`, interpolation/quantization discrepancy and extra `R_warp`. Never set all terms to zero or force tiny Blade covariance to pretend perfect temporal interpolation.
- Use Joseph-equivalent or independent square-root numerical oracle for current state covariance, symmetry and SPD certification. For unobserved points, exact prediction-only covariance; source anchor correlated with source depth once.
- Keep exact `6x6` per-point output and original point IDs; no zero-filled unpublished trajectories. Keep existing NaN tails, journal and capacity/process safety.
- Preservation: no edits to `legacy`, `simplified`, `full` models beyond surgical shared renderer bug fixes covered by regression evidence. Do not write placeholder tests that just check presence of classes.

## 6. Mandatory tests

**A. Analytical:** finite differences for all new target-sampling Jacobians wrt 6D current/reference states for multiple cameras, positive/negative displacements, nonzero source projection offset, source-depth change, bilinear cell transitions. Compare to independent NumPy oracle. Verify velocity derivatives zero in an instantaneous residual.

**B. Sensor/identity:** one-point dynamic translations ±1.3 pixels and ±3 pixels, 3D/2D motion with and without depth displacement, rotation-induced pixel displacement (independent points, not rigid estimation), color/depth edge, occlusion/reappearance, 2-camera handover, masks hard, current owner changes but correct identity preserved.

**C. Acceptance regression:** create a fixture where true point moves from integer current pixel `p` to `p+1` or more. Old fixed pixel must reject; new material-following must accept **only if the actual current observation supports it**. Show no false background updates and no objective improvement by deleting a measurement. Static object remains static.

**D. Local Gaussian numerical:** compare local state mean and 6x6 covariance to dense exact within-point augmented Gaussian/QR, including shared RGBD raw pixels, same-point temporal state cross-covariance, multiple same-camera references as appropriate, reference replacement and backwards time.

**E. Motion observability:** textured region -> recover true displacement; textureless aperture -> explicitly rank-deficient/confidence-limited and no invented velocity; photometric mismatch -> appropriately diagnosed. Measure condition number with physically interpreted measurement covariance; don't add made-up thresholds.

**F. Real SAPIEN:** first separate GT-falsification audit on USB/Blade frame 0→1 and 1→2. Then inference **without GT** with full 8 cameras and unchanged point identities. Posthoc evaluate moving-vs-static error and NEES after inference. Compare `legacy`, `simplified`, `local-correlated`, `material-following` on same points, masks, prior and source. Report pre/post pixel shift and % accepted >1 pixel motions, moving-point median/p90/p99 3D error, static p90 drift, point visibility, rejection causes, texture/observability statistics. No blanket claim of success from aggregate static medians or decreasing objective.

**G. Profiling:** true full-cloud 3-frame pilots; runtime/frame, max CUDA alloc, full cloud vs seeded-uniform factors 1/2/4/8 with correctly forwarded flags. Different point counts change ownership; disclose that. CPU oracle small; CUDA batched inference must not call global cuDSS full mode.

## 7. Required deliverables and failure classifications

1. A source-code diff and compact equation-to-code map (file:function:line), with a separate `material-following` mode and verified no use of old fixed-current row signature.
2. `reports/temporal_dynamic_gt_warp_audit.md` + JSON/PNG or HTML showing GT-only old vs new residuals, camera validity and depth-model bias.
3. `reports/temporal_dynamic_point_recovery.md` + machine-readable results, actual completed frame journals, per-point RGBD projections/associations and static-vs-moving error tables. Report `DYNAMIC_NOT_RECOVERED` if so.
4. Regression tests, finite-difference oracles, covariance parity, all originally passing tests; if previously failing archive tests exist, list separately, do not silently rewrite historical expectations.
5. Clear failure states: `GT_CORRESPONDENCE_UNVERIFIED`, `MATERIAL_WARP_INVALID_AT_GT`, `IMAGE_MOTION_UNOBSERVABLE`, `DYNAMIC_ASSOCIATION_FAILED`, `COVARIANCE_INCONSISTENT`, `DYNAMIC_NOT_RECOVERED`.
6. Exact commands matching the implemented CLI after tests. Do NOT launch 60-frame or ten-scene experiments until actual USB and Blade frame-2 dynamic points move toward their GT positions, without significant new static drift.

## 8. Commands to support *after* implementation

### CPU/CUDA unit tests

```bash
cd ~/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B -m unittest -v \
    tests.test_temporal_material_warp \
    tests.test_temporal_material_association \
    tests.test_temporal_material_local_covariance \
    tests.test_temporal_material_motion
python -B tools/test_temporal_rgbd.py
```

### GT-only physical falsification (diagnostic; no inference and no GT used inside inference)

```bash
cd ~/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/audit_temporal_dynamic_correspondence.py \
  --scenes USB_100109,blade_103706 \
  --frames 0,1,2 --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/material_warp_gt_audit"
```

### Actual local IEKF, full cloud, 3-frame gate

```bash
cd ~/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
bash tools/run_temporal_rgbd_critical.sh \
  --mode material-following --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --stop-after free \
  --temporal-runtime-budget-seconds 600
```

### Optional faster diagnostic (not equivalent visibility / not substitute for full-cloud gate)

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode material-following --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 8 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --stop-after free \
  --temporal-runtime-budget-seconds 600
```

**All new CLI flags, modes, modules and tests above are proposed work. Verify existence before executing.** The `ideal` warp setting is an explicitly *uncalibrated diagnostic assumption* and must never be called physically validated. The approved sensor covariance remains active. Do not invent a nonzero warp discrepancy merely to force acceptable residuals.
