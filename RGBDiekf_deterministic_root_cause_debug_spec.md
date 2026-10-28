# RGBDiekf — deterministic root-cause investigation of free-point explosions

**Scope:** F1/B1 independent projective RGB-D IEKF. Investigate only; do **not** add or change gates, χ² acceptance rules, velocity/displacement limits, artificial damping, static labels, regularization, correspondence matching, or optimization behavior. Do not modify F2/B2, coherence or joint fitting during this diagnosis. Any alternate measurement terms are to be evaluated **offline**, not substituted into production updates.

**Source version:** Verified public `SamueleMara/RGBDiekf` main `29a23a722aa98de137453717038054217a9ee5cb` (2026-10-28). Codex's subsequent changes (`row_innovation_gate`, `update_acceptance`, `scene/noise_calibration.py`, etc.) are described in its log but are **not committed** to that main branch. Before instrumenting the actual workspace, record `git rev-parse HEAD`, `git status --short`, `git diff --stat`, source hashes, arguments, and the full noise/policy manifests. Newer dirty-tree line numbers may differ; find functions by name.

## Verified evidence and what remains unproven

- USB, stapler, storage: each of six forward/backward full-debug runs contains points with metre-scale displacement. The largest correction for runaway displayed points occurs on an observed frame; the late trajectory is prediction-only and fits the constant-velocity transition at < 1e-6 m. This establishes the *accepted correction -> false velocity -> observation loss -> ballistic escape* sequence.
- **Storage F1 point 171031**, first divergent update frame 1, `||delta x||=0.2307m`, `||v+||=13.358m/s`, following frame zero accepted cameras, endpoint displacement ~13.36m. Reconstructed predicted position standard deviations 17–20mm, prior position Mahalanobis norm ~13.47 (squared ~181.4). The covariance coupling predicts the stored velocity to within ~2e-6m/s.
- **USB F1 point 81029**, first divergent update frame 2, `||delta x||~0.576m`, `||v+||~20.17m/s`, thereafter prediction-only. Earlier sample of archived RGB images showed large cross-camera appearance disagreement at the projected position; archived depth *color maps* permit only approximate depth differences, not an exact per-row depth-residual proof.
- Storage point 171031's source color was approximately `(0.118, 0.071, 0.071)`, while cross-camera samples at nearly the same canonical XYZ were `(0.643,0.549,0.407)` and `(0.729,0.631,0.451)`. The source is near an intense pixel-color transition. This proves that **the fixed single-pixel RGB and cross-view appearances disagree**. It does **not** prove those cameras' rows were used in the decisive IEKF iteration; that is precisely what the trace must establish.
- The earlier uploaded stand-alone HTML viewers contain no embedded trajectories; they require `debug/data/*.js`. `debug/img/depth_*.png` is a color map, **not metric depth**. Exact IEKF replays require the original `depth_npy/` and camera metadata, or a lossless per-row dump of the original measurements.

## Problems / precise original-code locations

| ID | File and original lines | Question to resolve numerically |
|---|---|---|
| M1 | `scene/endpoint.py:58-63`, `estimator/free_projective_run.py:74-79` | One source pixel defines invariant RGB and world XYZ: is cross-camera view/occlusion/edge appearance compatible with that model? Is source covariance accurate? |
| M2 | `estimator/projective_sensor.py:86-126`, notably `98-111, 112-125` | Valid 2x2 mask/depth support does not establish physical point identity. What are the exact raw pixel values, bilinear gradients, variance and image-edge sensitivity? |
| M3 | `estimator/projective_terms.py:43-62`, notably `53-61` | Which camera and channel contributes excessive `H^T R^-1 r`? How much is wrong-surface depth versus incompatible color? The original depth visibility policy is one-sided. |
| M4 | `estimator/projective_visibility.py:20-58,67-76`; `estimator/point_kalman_cuda.py:154-163` | Does frozen frontmost-pixel assignment remain valid as the iterate moves? Which per-camera statuses change between prior and later iterations? |
| K1 | `estimator/point_kalman_cuda.py:200-245` | Does each 6x6 solve match an independent dense oracle? Is the huge state correction a correct solution to a false likelihood, or a Jacobian/linearization/implementation error? |
| K2 | `estimator/point_kalman_cuda.py:246-253` | Does the filter accept a state despite nonconvergence or loss of consistency? Is `P+` evaluated at the accepted state/row set? |
| K3 | `estimator/point_kalman_cuda.py:38-53,251-253` | Does every prediction-only state satisfy exactly `x'=x+dt v`, `v'=v` and `P'=FPF^T+Q`, including signed backward `dt`? Expected ballistic behavior is not itself a bug. |
| C1 | `configs/example_uncalibrated.json:97-129,186-234,300-325`, `scene/config.py:42-56`, `estimator/free_projective_run.py:74-79` | Which `Pxx`, `Pvv`, `Pxv`, `Qc`, `Rrgb`, `Rdepth` are actually used? Default initial velocity component sigma is 1 m/s, `Qc` is 0.01 m²/s³ per axis, RGB variance .0025, depth variance 1e-4m²; all illustrative, not physically calibrated. |
| C2 | `configs/projective_rgbd.json:9-13`, `estimator/projective_sensor.py:117-125` | RGB/depth model-mismatch variance is zero on original main; how does cross-view residual distribution depend on camera pair, image gradient and viewing angle? Does a shared Gaussian `R` describe it? |
| N1 | `estimator/projective_sensor.py:98-125`, `estimator/point_kalman_cuda.py:180-189` | Invalid NaN/Inf can poison weighted sums when `0 * NaN`; Codex reports fixing this in its dirty tree. Confirm finite terms and equal results with masked valid rows. Not established as origin of finite runaway states. |
| D1 | `estimator/free_projective_run.py:62-64,107-116`, `point_kalman_cuda.py:264-305`, `train.py:88-92` | Current trace samples few point IDs and frames; does it save every row, every iterate and exact process/measurement matrices at the first divergent event? |
| Downstream (separate) | `estimator/projective_coherence.py:26-60`, `estimator/rigid_kalman_cuda.py`, `joint_reduction/joint_projective.py:52-94,169-184` | Contaminated free trajectories degrade core selection; pose and joint covariance adequacy later can fail independently. Do not confuse these with the initial runaway trigger. |

## Mandatory per-point, per-frame, per-iteration trace

For a single point *before it first departs*, record full `float64` values in an isolated JSON/NPZ, not rounded aggregate quantiles:

1. **Reproducibility and identity:** git SHA + dirty patch hash, exact scene/anchor, source camera and integer pixel ID, original XYZ/RGB, timestamps `t[k]`, signed `dt`, all camera K/Tcw, mask/depth/RGB input hashes, parameter provenance, original point ID and its subsample mapping, the current update policy (including Codex's added policy keys if present).
2. **Prediction:** `s_prev, P_prev`, `F, Q`, `s_minus, P_minus`, separately `Pxx`, `Pxv`, `Pvx`, `Pvv`, eigenvalues and condition number. Recompute with independent NumPy. Also compute the *unmodified* Mahalanobis displacement of any accepted `x_plus` from `x_minus`.
3. **For every IEKF iterate j:** `s_j`, `s_j - s_minus`, all projected `(u,v,Z)`; pre- and post-visibility reason per camera; nearest pixel and pixel-owner ID; four actual RGB and depth neighbors, 2x2 mask and depth validity, bilinear weights, interpolated RGB/depth, RGB/depth image gradients, canonical RGB, full 4-vector residual `r`, complete 4x3 `Hx`, `Rdiag`, the depth `Szz`. Record even rejected candidate rows with a numerical reason, but never feed them to the solver merely for debugging.
4. **Contribution breakdown:** For each currently admitted camera c and channel a, save `g_ca=H_ca^T r_ca/R_ca`, `I_ca=H_ca^T H_ca/R_ca`, and `r_ca²/R_ca`; split RGB and depth. Verify `Ixx=sum I_ca`, `gx=sum g_ca`. Save `Lam=(P_minus)^-1`, `A=Lam + I`, `b= -Lam(s_j-s_minus) + [gx,0]`, `delta=solve(A,b)`, `delta_x,delta_v`, norm/step and full per-iteration convergence signal.
5. **Nonlinear replay:** At `s_j` and proposed `s_{j+1}`, reproject the *same measured pixels/rows* and compute the true nonlinear residuals and prior penalty. Separately list changed/missing/added row IDs. Do not compare changing-size objectives as though their absolute sums were comparable. Save objective components for the common row set and all valid row sets separately. Log whether the IEKF decreases or increases the actual measurement objective and whether the projected surface changes.
6. **Final acceptance and covariance:** `s_final_iter`, `s_plus`, `P_plus`, `converged`, `observed`, `nrows`, reason codes, new Codex acceptance categories if applicable, posterior information spectrum and dense Joseph check with exactly the accepted final measurements.
7. **Next-frame propagation:** first later frame with no observations, its per-camera reasons and exact `s_minus`, proof of `x(t)=x_last + (t-t_last)v_last` and unchanged `v` on every remaining prediction-only step.

If a raw data array is unavailable in an old debug ZIP, report **unavailable** rather than inferring metric depth from the colorized visualization.

## Independent numerical oracles (debug only)

The filter computes `s=[x,v]`, `H=[Hx,0]` and

`A_j = Lambda_minus + sum_l H_l^T R_l^-1 H_l`,

`b_j = -Lambda_minus(s_j-s_minus) + sum_l H_l^T R_l^-1 r_l`,

`delta_j = solve(A_j,b_j)`.

Recompute `A,b,delta` in NumPy float64 from the dumped terms. Verify signs with finite differences of the *same bilinear RGB-D interpolant*, including `H_depth = grad(D) Ju - Jz` and `r_depth=Zpred-Dmeas`. Verify `P+` versus dense Joseph using identical admitted rows. This tests algebra; a correct algebraic result **does not** prove correct physical measurement association.

For any position-only measurement update at a fixed prior, the exact Gaussian conditional relation is

`delta_v = P_vx^- (P_xx^-)^-1 delta_x`.

Recompute this identity from the saved prior and accepted position correction. Storage point 171031 previously matched within approximately `2e-6m/s`; a new trace should demonstrate it without inference.

Compute `C_prior=(splus-sminus)^T Lam(splus-sminus)` and separately normalized RGB/depth data costs at the prior and at the candidate state. For storage 171031 the previously calculated 3D position-prior squared Mahalanobis was approximately **181.4**, yet the point moved. Quantify the actual photometric/depth cost improvement that overcame this prior penalty, or determine that the nonlinear objective did not improve at all.

Analyze identical-state, same-time cross-camera measurements at the canonical endpoint. Since `x` is the same instant, systematic RGB disagreement cannot be explained by genuine *temporal* object motion; possible mechanisms remain occlusion, border mixing, view-dependent reflectance, calibration or pixel correspondence error. Tabulate residuals by source camera, target camera, image-gradient magnitude, mask boundary, viewing angle, and depth discrepancy. Do not silently treat color differences as independent Gaussian noise.

## Offline factorial decomposition; no inference changes

With fixed recorded `s_minus,P_minus`, exact original pixels, camera geometry and per-iteration admitted row sets, replay in isolated numerical code:

- original full RGB + depth;
- RGB contribution alone;
- depth contribution alone;
- per-camera leave-one-out **as a diagnostic**, not as a new gate;
- fixed first-iterate row set versus naturally changing rows;
- original and Codex noise models **with the raw numerical values and decomposed sensor/model components reported separately**;
- original `Pvv`, `Qc` and alternative independently justified physical covariance values, one variable at a time, with resulting `Pxv`, gain and posterior displacement explicitly compared.

Do not deploy those replay variations as filter modifications. A point's source RGB, camera measurement and origin should remain constant across comparisons. Explain in terms of differences in objective, gradient, conditioning, covariance and measurement set—not in terms of whether a trial has fewer points above an arbitrary displacement threshold.

## Commands to reproduce the first original failures

Run on the **original code configuration** when proving the original error; Codex's uncommitted tree has added policies and `--sensor-noise`. First inspect `python train.py --help` and record all settings. Use original dataset timing and preserve full camera/point identities (subsample factor one). Avoid `--frame-limit` if it changes the original frame timestamps or point selection unexpectedly.

```bash
# Original first divergent event: storage F1, frame 1
python train.py \
  --scene "$STORAGE_SCENE" --output "$OUTROOT/storage_trace_171031" \
  --direction forward --stop-after free --frame-limit 3 \
  --pcd-subsample-factor 1 --covariance-precision float64 \
  --debug none --debug-frames 0,1,2 --debug-point-ids 171031 \
  --no-evaluate --no-qa

# Original second divergent event: USB F1, frame 2
python train.py \
  --scene "$USB_SCENE" --output "$OUTROOT/usb_trace_81029" \
  --direction forward --stop-after free --frame-limit 4 \
  --pcd-subsample-factor 1 --covariance-precision float64 \
  --debug none --debug-frames 0,1,2,3 --debug-point-ids 81029 \
  --no-evaluate --no-qa
```

The stock trace files are `free_forward/numeric/trace_frame_0001.json` and `trace_frame_0002.json`; stock tracing already includes some `state`, `A`, `b`, `delta`, row `r/H/R`, and final Kalman/Joseph checks, but does **not** provide the full raw-neighbor + prior/objective + per-channel attribution requested above. Instrument the existing `PointMeasurement.__call__`, `build_terms`, `_trace_iteration` and `_trace_final` for only the chosen IDs. Avoid expensive global snapshots. **Backwards caution:** backward canonical source is frame T; trimming the dataset to its first 3–4 frames changes the backward endpoint and invalidates a comparison. Use the full sequence or an exact tail slice preserving frame T.

## Interpreting the outcome (no gates)

- **Original solve != independent oracle:** numerical implementation / sign / accumulator / covariance bug. Fix the algebra after identifying the mismatch.
- **Solve matches oracle, but objective increases or rows jump across different physical surfaces:** nonlinear linearization and changing measurement-set inconsistency; report the exact step/cameras/pixels.
- **Solve matches, objective decreases, residual improvement is dominated by cross-view RGB on a sharp edge:** the assumed invariant point appearance / likelihood is wrong for that point.
- **Solve matches, depth dominates and sampled range lies on another surface:** physical-surface identity/visibility model is wrong despite locally valid 2x2 interpolation.
- **Solve matches and sensor residuals are small, but `P/R` are empirically inconsistent:** investigate covariance units, cross-view correlations, and calibration, not blanket motion suppression.
- **Large velocity appears without position/velocity coupling identity:** Kalman covariance/state bug; check `F/Q` and update code.
- **Ballistic tail does not equal exact signed-timestep propagation:** prediction implementation or stored-indexing bug. Prior archives showed the tail was exact, so this is mainly a regression check.

**Deliverable requested from Codex:** one source- and dataset-hash-reproducible report per failure with exact first divergent step, raw camera measurements, full 6x6 prior/update/posterior, per-camera/channel information contribution, actual nonlinear objective versus previous iterate, row-set changes, independent NumPy oracle comparison, and proof/negation of each causal hypothesis. An uncertainty-qualified root-cause conclusion; **no new threshold or filter modification in this task**.
