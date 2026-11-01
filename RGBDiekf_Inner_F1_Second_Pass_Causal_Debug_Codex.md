# CODEX INVESTIGATION — F1 STATIC DRIFT, SECOND PASS: PROVE THE INITIATING CAUSE

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Mode:** **Investigation only. Do not edit the inference path. Do not run RTS, IEKS, EM, model selection, or another complete scene.**

## 0. Mission and stopping rule

The first report (`reports/f1_static_drift_forensics_20261030.md`) convincingly reproduces the standard F1 IEKF arithmetic and proves the position-to-velocity *amplifier*, but **does not yet establish the initial cause of the correction on every failing point**. "Depth is ambiguous," "P_vv is too large," or "the code agrees with its equations" are insufficient final explanations. This second pass must identify the precise observation, residual, derivative, model mismatch, admission/visibility decision, or nonlinear numerical event that produces the **first accepted incorrect correction**. Quantify its necessity and sufficiency with controlled *single-point/single-frame* replay.

**Critical newly identified fact:** In `estimator/free_projective_run.py::run_free`, F1 initializes from frame-0 source-camera depth, calls `_store(..., k0, ...)`, and enters the update loop with `for ... in zip(order, order[1:])`. **It never assimilates the independent non-source cameras at frame 0.** A constant intercamera discrepancy therefore first appears to the filter at frame 1, after the velocity process prior has enlarged position uncertainty. Determine whether this is the principal initiating mechanism.

**DO NOT** claim success until, for each mandatory case, you show (i) the original accepted update reconstructed to numerical precision, (ii) the raw frame-0 and frame-1 observations at the fixed original point, (iii) the actual source of the initial gradient/information, (iv) a minimally invasive counterfactual that isolates it, and (v) a numerical explanation of accepted velocity and its later propagation. Distinguish **demonstrated cause**, **necessary contributing factor**, **correlation**, and **unobservable/undecidable**.

## 1. Strict operational constraints

1. Use existing full-scene standard F1 arrays, traced runs, and raw input frames only. The first forensic report lists paths under `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/`. Do not overwrite them. No full-scene filtering, no full-scene re-evaluation, no 100k/300k benchmarks, and no full Laptop run.
2. Prefer **USB_100109 and Storage** because their failure modes differ. Blade only for the already-reported depth-gradient example; don't run a new Blade sequence. Limit experiments to selected original point IDs, their decisive frames and original camera images. Small CPU/GPU single-point micro-replays are allowed; rebuild *global frontmost pixel ownership* from the saved whole-cloud state at one frame if necessary, but do not recompute a whole sequence.
3. No modifications to `train.py`, `estimator/point_kalman_cuda.py`, `estimator/projective_sensor.py`, `estimator/projective_terms.py`, `estimator/projective_visibility.py`, default configs, inference flags, noise/prior parameters, or anything reachable by production inference. Make new diagnostics under `tools/` and small independent tests under `tests/` only. Read-only counterfactuals must never write baseline outputs.
4. No GT for estimation/selection of an improved step, noise adjustment, or likelihood. GT and synthetic mesh/part geometry can be used **only in diagnostic auditing**, with clearly labeled oracle-only evaluations. Do not equate part labels with material-point identity; sparse static GT is not exact point GT.
5. Numeric comparisons must name the baseline file, commit/source hashes, frame, camera, original point ID, measurement origin, exact row/cell, and scalar/vector units. Verify trace parity to previous outputs; no new large trace run. If trace data are insufficient, replay **one point and one frame** with the stored full-scene state, without advancing the full filter.

## 2. Priority points and matched controls

Reproduce, at least:

| Point ID | Scene | Failure to explain |
|---|---|---|
| 8444 | Storage | Frame-1 source camera gives zero residual; non-source camera color ~1–1.6 sigma yields 24.1 mm movement and 1.41 m/s velocity; ends 1.31 m away |
| 49199 | Storage | Similar apparent noise-level RGB-driven correction; 10 nonconverged IEKF iterations |
| 67827 | USB | Frame-1 crease; ~−4.45 mm depth (=−80 sigma_R), predicted S gate permits it |
| 30502 | USB | First accepted rows at frame 3, depth crease; iterative state moves to far face; neighboring point 30503 stable |
| 51975 | USB | Quantized-color driven drift; neighboring point 51974 stable |
| 262894 | Storage | Occluders 30.9/67.2 mm ahead pass initial gate, then rows change and final correction mainly color |
| 10251 | Blade | Apparently consistent depth-only observation; inverse depth reduces spurious tangential correction |
| 14104 | Blade | Nonconverged 30.6 mm correction, loses rows permanently |
| 67801, 102354 | Storage | Known accepted states with *increased fixed-support objective* |
| 8416–8418 | Storage | Source-pixel gradient discontinuity creates up to 0.79 m oracle divergence |
| 87290, 85696 | USB | Moving-part support collapse; distinguish prediction-only from wrong reassociation |

Also include matched stable controls from `selected_point_ids.csv`, including USB 51974 and 30503 when valid. Do not assume these are guaranteed one-to-one physical neighbors.

## 3. CRITICAL NEW TEST: frame-0 versus frame-1 multiview evidence

For every selected static point i and each camera c, **at the fixed original world position** x_ref_i (no IEKF updates), evaluate the full raw RGB-D samples at **both frames t=0 and t=1**, using each original camera's fixed calibration and the exact production sampler in read-only mode. Include source camera and all non-source cameras. If the point is occluded in either frame, label it; do not compare an invalid sample as though it were an observation.

Compute the residual vector with fixed source color a_i:

    r_{c,t}(x_ref) = [a_i - I_{c,t}(pi_c(x_ref)); Z_c(x_ref) - D_{c,t}(pi_c(x_ref))].

**Most important decomposition:**

    r_{c,1}(x_ref) = r_{c,0}(x_ref) + Delta_r_{c,1:0}(x_ref).

Since x_ref and cameras are fixed, the shared source color cancels in the difference:

    Delta_r_RGB = I_{c,0}(pi_c(x_ref)) - I_{c,1}(pi_c(x_ref));
    Delta_r_depth = D_{c,0}(pi_c(x_ref)) - D_{c,1}(pi_c(x_ref)).

Interpretation:

- If r_{c,0} is large and Delta_r approximately zero, the error already existed **before the first update** and supplies no measured evidence of temporal motion. Determine its origin: source-point geometry, camera calibration, view-dependent appearance, interpolation/quantization, or occlusion.
- If r_{c,0} is small and Delta_r is large, inspect a genuine temporal change: moving-object occlusion, illumination/shadow, depth-map changes, camera pose inconsistency or other acquisition artifact.
- If both differ, quantify contributions; do not attribute the entire frame-1 innovation to motion.

**DO NOT** claim that any nonzero r_0 must be geometry error. A constant per-view RGB bias can produce the same pattern. Identify what can and cannot be distinguished from the available data.

Required outputs per (point, camera):

- original frame-0 and frame-1 source RGB/depth pixels and 2x2 neighborhoods; exact u,v, subpixel fractions, masks, raw_valid, visibility and owner IDs;
- raw and whitened 4D r_0, r_1, Delta_r; per-channel source-color and RGB interpolation precision; H_x at both frames; rank, singular values and source of deterministic differences;
- frame-0 raw support reason and source-camera-duplicate bookkeeping; count independent non-source valid observations that were ignored at t=0;
- frame-by-frame 0..5 residual at **x_ref held constant**, not the evolving track, in each camera, plus optional later sampled frames where visibility remains physically valid;
- full original RGB/depth 15x15 image crops centered on pi_c(x_ref) for relevant cameras at t0/t1; show a marked subpixel location and original 2x2 support, with comparable scales.

**Mandatory counterfactual CF0 (not a production change):** For a chosen point, solve the original exact linearized 6D Gaussian update in two hypothetical schedules using exactly the *same data provenance*: (A) source-camera anchor only at t0 then non-source cameras first at t1 (production); (B) source anchor plus all independent, validated non-source camera rows at t0, then frame-1 rows. Do not double count the source depth/color. Reuse the same P0, R, Q, and original RGB-D data. State clearly when nonlinear association differs. Compare predicted state, x0 correction, v1, and objective. This is a diagnostic **causal counterfactual**, not authorization to modify F1 or proof that t0 fusion fixes all biases.

## 4. Explain the precise first-position correction, not merely the Kalman velocity gain

For each first erroneous update, reconstruct from exact production matrices:

    Lambda = (P_minus)^(-1)
    F_i = sum_{accepted rows r} H_r^T R_r^(-1) H_r    [only position 3x3 block]
    g_i = sum H_r^T R_r^(-1) r_r
    (Lambda + diag(F_i,0)) Delta_s = [g_i,0] - Lambda(s_iter - s_minus).

Log **each inner j**, not just initial and accepted:

- s_iter, s_minus, P_minus full 6x6, A, b, Delta_s, step, step metric, accepted/fallback/converged flag, and exact iteration chosen;
- for every camera: raw r_RGB/r_D, four sampled pixels, H_RGB/H_D, R, S, NIS, per-camera contribution g_r=H_r^T R_r^-1 r_r and Hessian/Fisher H_r^T R_r^-1 H_r; reject reason before/after;
- camera-support bitset and 2x2 cell ID at *every iteration*, with per-channel objective at that iterate;
- delta_x, delta_v and verify Delta_v = P_vx^- (P_xx^-)^-1 Delta_x where appropriate. Distinguish this **exact amplifier** from the cause of Delta_x.

### 4a. Eigenvector and surface decomposition

Compute the eigendecomposition of the **3x3 effective position information** using the actual 6D prior (Schur complement if appropriate). Print eigenvalues/singular values, condition number, and components of the accepted position step along each information eigenvector. Do not confuse high depth information in the normal with actual tangential observability. Obtain a local depth-surface normal from an independent fitted plane only if the 3x3 patch is demonstrably single-surface; at a crease label normal undefined/multiple. Report projections Delta_x_normal and Delta_x_tangent, and per-channel gradient projections onto those axes. A point on a uniform plane may be tangentially unobservable; that is a statement about **information rank**, not proof that a particular accepted 24 mm correction is the correct solution.

### 4b. Explain why 1-sigma residual gives a 24-mm step

For Storage 8444, calculate the actual scalar components of g_RGB, g_D and the effective inverse information along the accepted direction. Produce a numerical equality of the form

    (effective information matrix)^(-1) * (effective gradient) = observed displacement

and show which camera/channel provides the gradient. Then compute the same vector at x_ref using frame-0 images; is the gradient essentially unchanged? If yes, it is not independent frame-1 motion evidence. Show whether the accepted position is a local minimum, an iterate-cap artifact, or a support-loss artifact.

## 5. TEST FOR THE SECOND MISSING MECHANISM: acceptance by dropping observations

The standard F1 code `point_kalman_cuda.py::PointMeasurement.__call__` reprojects and resamples at every iteration; `update_points` accepts candidate states by a chi-square adequacy check using **the rows valid at the candidate** and a threshold indexed by the **candidate row count**. When a previously conflicting row drops out, both the residual sum and degrees of freedom change. That can make an iteration appear acceptable without improving the original evidence.

For each j and the final accepted/fallback state, report:

1. `S_prior`: camera IDs/row identities usable at the original predicted state;
2. `S_j`: camera IDs/row identities usable at iterate j; and `S_final`;
3. transitions `VALID -> INVALID` and `INVALID -> VALID`, exact reason including mask, 2x2 discontinuity, front-occlusion, chi2 gate, duplicate ownership, image border;
4. old and new pixel-cell IDs even when camera membership stays constant;
5. `J_prior` on the original valid rows; `J_j` on the same rows wherever physically evaluable; `J_j` on its own rows; normalized row-count-dependent acceptance threshold;
6. original-row residuals evaluated at the candidate **for audit only** even when current gate would reject them. If pixels are outside mask/depth valid set, mark 'not evaluable'; never fill with zeros.

Apply to Storage 8444, 49199, USB 30502, USB 67827, and the documented cost-increase cases 67801/102354. Produce a distinct verdict:

- **Model preference on unchanged rows** (actually lower nonlinear objective),
- **row-drop advantage** (conflicting observations disappeared),
- **pixel-cell change advantage** (different surface interpolant/sample),
- **nonconverged iteration artifact**, or combination.

A χ² adequacy test on an iteration's own rows is **not** a monotonic objective guarantee. Do not call it one.

## 6. Finite, controlled single-point counterfactuals (diagnostic only)

Use the original same-frame predicted state, original P, R, and *unchanged other-point occupancy* for all arms. Re-run only the point's local update and keep raw matrices and objective traces. No GT-dependent tuning or new production policy.

| Arm | Change in the replay | What it discriminates |
|---|---|---|
| A | Exact baseline micro-replay | Reference: reproduce all j and accepted state |
| B | Freeze the original camera-row membership while evaluating the actual nonlinear sample at every trial; explicitly signal invalid support | Does row removal cause apparent acceptance? |
| C | Include independent validated non-source frame-0 observations before frame 1; never reuse source RGB/depth independently | Constant cross-view mismatch mistaken for motion? |
| D | Drop **only the suspected camera**, then replay; repeat one camera at a time, always report row-count and information changes | Which observation is necessary for the correction? |
| E | RGB-only and depth-only local solves, using the same original support and consistent A (not post-hoc additive deltas) | Which channel actually generates the accepted step? |
| F | Keep all observations but use exact bilinear derivatives from the left/right sides of integer-pixel boundaries; also perturb u,v by ±1e-12, ±1e-10, ±1e-8 pixels | Is the nonlinear iteration expanding at a nondifferentiable cell boundary? |
| G | For one step **only**, zero the position-velocity cross-covariance or condition v=0, without altering the measurement rows | Does this remove velocity injection while leaving spatial misfit? A diagnostic, **not** a proposed velocity prior |
| H | Hold the point physically at x_ref and compute all eligible per-camera r_t across t=0..5 | Is the alleged motion 'evidence' actually constant in time? |

For each arm, save actual objective, original-row objective, support changes, accepted Delta_x/Delta_v and an explanatory **causal prediction**. A null result is informative; report it.

## 7. Prove the nonlinear optimizer's actual landscape

For at least Storage 8444, USB 67827 and one knife-edge case, sample the **actual non-linear per-camera RGB-D functions** along:

    x(alpha) = x_minus + alpha*(x_accepted - x_minus)

for alpha in [0, 1] plus several interior points, and extend modestly around endpoints. Keep the appropriate v(alpha) consistent with the corresponding 6D update path; report prior energy exactly. At every alpha record all per-camera residuals, 2x2 cell, validity, chi2, and support. Plot:

- prior cost, RGB cost, depth cost and total cost using **each candidate's own rows**;
- same costs on **a stated common set of physically evaluable rows**;
- first-order linearized model predicted decrease versus actual measured decrease;
- vertical marks where pixel cell, depth validity or gate status changes.

If the path crosses an invalid cell, draw a gap/invalid segment; do not invent an objective there. Analyze whether accepted x is truly a lower-energy position on the same observations, or whether the 'improvement' is an effect of dropping measurements and discontinuities.

For knife-edge cells, independently compute one-sided Jacobians and the derivative of the one-sided accepted iteration map, or use a small directional perturbation test; report the amplification factor of perturbations per iteration. A reproducible 0.79 m difference due to a 1e-14 pixel perturbation is a **severe numerical stability failure** even if neither floating point implementation violates the algebra.

## 8. Moving-point disappearance: identify the exact **last usable camera**

For USB moving points 87290 and 85696, and one matched correctly tracked control, inspect the precise frame of first all-camera loss and the immediately preceding successful frame. For every camera at both frames, report:

- projected u,v,Z at predicted and GT-driven diagnostic locations, original mask, raw depth, whether a 2x2 sample exists;
- **all** individual gate reason codes (avoid collapsing to a single priority-coded reason), exact measured depth difference, S_DD and NIS, pixel winner/runner-up id and their true/estimated locations;
- whether the camera would provide a valid row under read-only `per-iteration` occupancy versus the default frozen per-frame assignment;
- whether the point was physically visible in the synthetic scene (if actual ray/material/mesh oracle is available), otherwise `cannot_establish_visibility`;
- actual velocity, predicted displacement, true moving-part displacement; whether the point tracks or coasts.

The previously reported '98% have duplicate suppression in at least one camera' is **not proof** that duplicate suppression caused all-camera loss. Establish which mechanism rejects the **last** surviving camera. If several independent reasons coexist, quantify them without falsely assigning a unique cause.

## 9. Source geometry, measurement model, and synthetic provenance

Verify, without modifying calibration:

- x_ref was back-projected from source camera integer pixel and positive metric Z; its source projection residual at t0 is zero to the expected precision;
- camera poses/intrinsics and frame/image indexing are fixed and consistent across t0 and t1;
- synthetic static pixels in each camera genuinely remain constant if the relevant ray continues to image the same static surface; inspect shadow, shading, occlusion, anti-aliasing and quantization effects;
- reported `R` is sensor noise plus interpolation and declared model mismatch, not an error bound for cross-camera appearance consistency, unmodeled depth discontinuity or material identity;
- frame-0 source-depth uncertainty was used once in P0; do not count it again in a hypothetical t0 update.

Clearly separate (a) uncertainty due to noisy pixel location/depth, (b) systematic intercamera residual, (c) actual temporal image change, (d) wrong-surface rows, and (e) unobservable tangent motion. The root-cause statement must name which are active for each point.

## 10. Proof requirements / acceptance gates

**Gate G0 (parity):** baseline behavior unchanged and representative point micro-replay matches original accepted state and row/status history; otherwise diagnose the replay itself before interpreting any counterfactual.

**G1 (frame0-time):** for every selected static point with an accepted false velocity, display per-camera r0 and r1 at the fixed x_ref, and their difference, with valid/occlusion status. This is mandatory even if r0≈r1 is not observed.

**G2 (first correction):** reconstruct the 6x6 linear algebra from the raw saved H, R, P and residuals; attribute g and eigenspace displacement to actual cameras/channels. 'Velocity = 59×position' alone does not pass.

**G3 (acceptance):** identify all row/cell changes per j; compare own-support and same-support objectives and reason about missing factors. Clearly show whether the accepted candidate improved the original physical evidence.

**G4 (causal intervention):** at least two independent counterfactuals must materially discriminate competing hypotheses for each of Storage 8444 and USB 67827. Include the **frame0 non-source-camera** counterfactual and **camera removal/frozen-support** where relevant. Record negative evidence.

**G5 (moving support):** identify the last viable camera and exact reason(s) for all-camera row loss for USB 87290/85696. Show predicted-vs-true motion separately.

**G6 (numerics):** provide a step-by-step divergence trace for the integer-pixel boundary case and the fixed/changed support cases. No claim of stability if accepted states depend catastrophically on floating-point rounding.

If any essential evidence requires unavailable surface/material GT, write `UNPROVEN: material identity` rather than pretending sparse part labels establish it.

## 11. Outputs — minimal, auditable, no giant files

Write to a fresh, clearly named subfolder under existing forensics root (do not modify old artifacts):

- `SECOND_PASS_ROOT_CAUSE.md` — a **point-by-point causal explanation**, sections for every mandatory ID; exact camera/frame/iteration, source of discrepancy, full 6x6 gain, accepted result and outcome; a final evidence ranking, with caveats;
- `frame0_frame1_raw_residuals.csv` — ALL cameras, source/non-source, raw/whitened residual, delta, support;
- `first_update_decomposition.csv` — each H/R/P/g/F contribution and eigenvector component; link to compact raw NPZ for exact arrays;
- `iteration_support_objectives.csv` — every j, actual vs common support, cells, reason codes, degrees of freedom, acceptance;
- `local_counterfactuals.csv` — one row per arm and point; input hashes, resulting correction, support and objective;
- `last_camera_loss.csv` — moving cases only;
- `objective_1D_<scene>_<id>.png` and `rgbd_patch_<scene>_<id>.png` — at least 8444, 67827, 8416;
- `tests/test_f1_second_pass_oracles.py` — small deterministic tests verifying frame0-vs-frame1 algebra, 6D solve identity, support bookkeeping and objective comparison.

The report must finish with an explicit statement for each key point:

> "This point first moves at frame __, because observation(s) from camera __ have a discrepancy of __ that [already existed at frame 0 / arose only at frame 1]. The exact contribution __ combined with effective information __ produces a position step __. The inner loop [retains/loses] rows __ and [decreases/does not decrease] the objective on comparable rows. The Kalman gain converts this step to velocity __; at subsequent frames __ observations remain and __ are lost, leading to final drift __. The controlled replay __ removes/does not remove the initiating correction. Remaining uncertainty: __."

If the result is still only a restatement of the Kalman equations, **the task is incomplete**. Do not propose fixes or retune parameters until the mechanisms above have been discriminated.
