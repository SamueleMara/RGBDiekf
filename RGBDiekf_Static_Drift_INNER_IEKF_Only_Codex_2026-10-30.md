# CODEX — RGBDiekf Phase-1 F1: forensic root-cause investigation of static-point drift

**Status:** INVESTIGATION ONLY — no estimator fix, no outer-loop development.  
**Scope:** the **STANDARD, DIRECTIONAL, FORWARD 6D IEKF (F1)** on **one synthetic scene: `USB_100109`**.  
**Objective:** establish a numerical, reproducible causal account of **why points belonging to the physically static part move**, and why moving-part points lose the object even though some points continue to its final configuration.  
**Explicit exclusion:** DO NOT RUN OR USE RTS / IEKS / temporal outer optimizers / EM / static–dynamic model selection / Phase 2 or Phase 3. We will investigate the outer loop **only after** the inner F1 has a demonstrated root cause and a verified correction.

## 0. Critical priority change — read first

Earlier investigations centered on the `temporal-ieks` output. **That is no longer the target estimator.** We already know that this outer loop makes static-point drift worse. We must now establish the cause of the drift **already present in the standard IEKF**.

- **Run the standard directional F1 IEKF, not `temporal-ieks`.** Inspect the actual local CLI before choosing arguments. Use the working command/configuration and scene from the existing directional baseline; if the CLI supports `--free-solver directional`, use it, otherwise select the equivalent documented default. Ensure the result's manifest explicitly identifies the **directional/standard solver**.
- **Do not** use `smooth_mean.npy`, `warm_start_smooth_mean.npy`, `objective_history.json` from IEKS, or the outer attribution's `A_onset_push_at_rest` and `B_objective_preference_final_rows` as if they were gradients or updates computed by the standard IEKF. They are useful *historical clues only*.
- **Exactly one synthetic scene:** `USB_100109`. Use all recorded frames (60, with actual timestamps), all calibrated cameras and exactly the original F1 policy; no other synthetic datasets, no real Laptop solve, no broad sweeps or full-scene benchmarks.
- Use the existing standard F1 result if complete and correctly attributable. Because the user explicitly wants the **standard IEKF run**, perform **one clean standard F1 run** with diagnostic instrumentation when necessary. If the existing run can be verified as bitwise identical and provides every needed state, it may be reused, but any missing per-iteration traces should be obtained by an **instrumented standard-IEKF run**, not by substituting IEKS traces.
- Do not change RGB-D noise, initial priors, process noise, dt, interpolation, gating, visibility, rejection policy, point sampling/subsampling, numerical precision, masks, or training/optimization settings while collecting baseline evidence.
- Add only **opt-in instrumentation** or separate read-only analysis scripts. Tracing OFF must leave the original F1 results bit-for-bit identical. The extra run must write to a **new distinct output root**; never overwrite an existing result.
- Synthetic ground truth may be used **after** estimation to select and evaluate points. It MUST NOT enter the F1 filter, measurement association, row gating, acceptance, initialization, covariance calibration, or trajectory optimization.

### Confirmed context; do not mistake these for the root cause

From earlier `USB_100109` logs, using a common static-part evaluation set:

| Estimator | Median final static displacement | Fraction of static points >10 cm |
|---|---:|---:|
| Original standard directional F1 | **13.55 mm** | **17.8%** |
| New temporal F1 forward warm-start | **10.76 mm** | **15.4%** |
| Outer IEKS result | **49.76 mm** | **33.7%** |

These are *different estimator paths*. Our target is the **original standard directional F1**, including its actual accepted IEKF iterations and its specific observation support. Do not interpret the 49.76 mm IEKS result as F1 drift.

The previously reported ~77% depth-dominant onset force and enormous final depth preference came from the **outer attribution calculation**; they are *hypotheses* for F1 until recomputed using the **standard IEKF's exact original rows, priors and accepted steps**. Depth may be responsible, but do not assume it.

## 1. Definition of a successful investigation

For **at least 3 badly drifting static point IDs**, **2 stable static controls**, and **2 moving-part points** (one successful, one lost), provide a causal timeline with concrete numerical evidence:

1. Exact initial backprojected point, source camera/pixel, covariance, and the point's static or moving part label (offline GT evaluation only).
2. Exact first frame and **inner IEKF iteration** with a nonzero accepted update that is inconsistent with physical staticness.
3. Raw 2×2 RGB-D samples **from each camera**, projection, visibility reason, row gate, depth interpolation, signed residual, gradient and covariance at that instant.
4. Exact **joint information matrix** / Kalman gain and RHS, reproduced independently in float64; decomposition into depth-vs-RGB effect; actual accepted 6D position and velocity correction.
5. The next predicted state and covariance; trace how the false velocity integrates, which cameras lose the point, and why the next accepted row does or does not belong to the correct surface.
6. Identify the earliest mechanism that **causally** produces appreciable false motion. Prove whether it is depth interpolation bias, wrong-surface depth/visibility, innovation gating, position–velocity coupling, an acceptance mistake, an incorrect Jacobian, or a combination.
7. Quantify how many moving-part IDs remain in the state but lose measurement support, attach to another surface, drift offscreen, or are actually deleted (do not assume disappearance means deletion).

Do not stop at aggregate statistics, qualitative plots, or changing a parameter until drift becomes smaller. **A smaller error after changing a knob is not a causal proof.** If a chain link cannot be measured, mark it **UNPROVEN** with the exact missing evidence.

## 2. Provenance and baseline execution

Inspect the **local working copy used to generate the data**, not only GitHub `main`; local experimental code may be uncommitted. Resolve the actual execution path in:

- `train.py`: stage selection, `--free-solver`, `--direction`, `--stop-after`, point sub-sampling and configuration.
- `estimator/free_projective_run.py`: `build_catalogue`, `run_free` and output files.
- `estimator/point_kalman_cuda.py`: `predict`, `PointMeasurement.__call__`, `update_points`, accepted-iterate logic, posterior covariance, per-iteration trace.
- `estimator/projective_sensor.py`: `sample` and `sample_cpu_oracle`.
- `estimator/projective_terms.py`: `build_terms`, residual `r`, Jacobian `H_x`, `S`, `nu`, NIS, row gate.
- `estimator/projective_visibility.py`: frontmost-point/pixel ownership, `apply_frozen_assignment`.
- `estimator/projective_camera.py`: `T_cw`, intrinsics, world-to-camera projection and Jacobians.
- `scene/endpoint.py`: canonical point backprojection and source uncertainty.
- `configs/projective_rgbd.json`, original noise configuration, exact calibrations/timestamps/masks.

Before any run, record original command, output path, source/hash/dirty status, Git commit, Python/Torch/CUDA versions, original full/subsampled point-ID map, frame and camera order, sensor-noise configuration, `P_v0`, `q_a`, depth variance and policy values.

**Run policy**

1. First audit the **existing standard directional baseline** on USB and determine whether it is genuinely the same F1 configuration used for comparison.
2. If it lacks essential accepted-step data, add **read-only opt-in tracing** to the **standard** `update_points` path. Instrument **only selected PIDs** and a small window around their first false updates; allow short expansion of those windows when causal events require it.
3. Launch at most **one new full USB standard F1 run** if required for authentic trace capture. Use exactly the baseline command/options, replacing only the output root and adding the opt-in trace IDs/options. Do not accidentally invoke the outer solver, or run the whole four-stage pipeline. Stop at F1.
4. Verify **tracing invariance**: disabled and enabled instrumentation must produce identical accepted states, observation decisions and covariances at the same precision; preferably hash original output arrays and compare them with the instrumented run. If not, stop and investigate tracer interference.
5. If a clean full run is not necessary, use **targeted replay of actual standard-IEKF frames** with the same prior, all other points needed for pixel ownership, and the actual full-image observations. Clearly note when a replay omits global visibility context and therefore cannot reproduce the run.
6. Never reuse a previous result that has different settings or point ordering as though it were a valid control. No Laptop run until the user authorizes it later.

Suggested preexisting baseline and historical IEKS paths **from the supplied logs**; verify on the actual machine before accessing:

```bash
AUD="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/static_equilibrium_audit_20261029_1718"
BASE="$AUD/USB_100109/baseline/free_forward"
# The temporal root below is HISTORY/CONTRAST ONLY. DO NOT RUN IT.
IEKS_HIST="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_phase1_20261029_2218/temporal_USB_100109"
```

**Do not invent or assume CLI switches.** Read `python train.py --help`, existing run manifest and actual baseline command; construct the one-scene F1 command from those facts. Save the command verbatim in `README.md`.

## 3. Select auditable point IDs based on STANDARD F1

Select about **24–36 distinct original point IDs**, deterministically, **from the standard F1 output**, not from IEKS final-error rankings:

- 8–10 static points with >10 cm final displacement under **standard F1**; sample varied onset frames/source cameras/depths and p50/p90/p99 severity.
- 5–6 static with 1–10 cm standard-F1 drift.
- 5–6 stable static <1 mm, matched to drifting cases in source camera, local depth slope, original projected subpixel fraction, number of visible cameras and image texture.
- 4–6 moving-part IDs, including those that remain on the articulated part and those that visibly disappear / become prediction-only.
- Optionally 2–4 points with missing observations from the outset, as negative controls.

Save `selected_point_ids.csv` with all ID mappings, source frame/camera/pixel, original XYZ, static/moving GT basis and caveats, F1 endpoint displacement and first event frames.

**GT caveat:** membership in a physically static part proves its material points should not move, but nearest-neighbor GT part labels may not give an exact match to the original RGB-D surface sample. Report **transport drift** `||x_t^+ - x_0^+||` separately from any truly matched GT localization error. Never silently call canonical deviation exact GT point error.

## 4. Trace the STANDARD F1 from frame 0, with signed physics

For each selected ID and every frame, record:

- `t_k`, `dt`, `s^-=[x^-,v^-]`, `s^+=[x^+,v^+]`, full predicted and posterior 6×6 covariances `P^-`, `P^+` including `P_vx`.
- Prediction-only vs accepted update; number of cameras/rows; per-camera invalidity reason and accepted-row identities.
- Inner iterations `j=0..n`: the exact candidate, computed correction, objective/statistical test, convergence, last-consistent-iterate fallback, **accepted iterate** and actual final `s^+`. Distinguish a candidate from a state actually accepted.
- First nonzero **accepted** `delta_x = x^+-x^-`, first `delta_v = v^+-v^-`, first increase in canonical drift, first camera/row lost, first re-admitted row, first exit from camera/foreground.
- Full frame-0 source-measurement provenance and whether source RGB/depth was correctly excluded from the frame-0 likelihood to prevent double use.

Check each frame numerically:

\[
 s^-_k=F_k s^+_{k-1},\quad
 P^-_k=F_k P^+_{k-1}F_k^\top+Q_k.
\]

For accepted updates, verify the identity

\[
 x^+_T-x^+_0
 =\sum_{k=1}^{T}\Delta t_k v^+_{k-1}
  +\sum_{k=1}^{T}(x^+_k-x^-_k).
\]

This decomposes **transported drift from integrated velocity** versus **direct subsequent corrections**. Include signed **vector** contributions, not merely ratios of norms. A high velocity-integration share identifies an amplifier, **not necessarily the initiating measurement error**.

## 5. Reconstruct every decisive measurement row exactly

For the first accepted false motion and the frames immediately before/after the first row loss or re-admission, extract all relevant camera measurements and log:

1. PID, frame/time, source-camera/pixel identity, projection `(u,v,Z)` and camera `T_cw`,`K`,`near`.
2. Nearest pixel used for frontmost ownership; floor-based 2×2 pixel coordinates; `(du,dv)`.
3. Four **raw depth values**, four RGBs, four validity flags and four foreground mask values; local depth min/max; exact same-surface threshold in metres; reason code.
4. Arithmetic interpolated depth, RGB and exact derivatives `∇D_uv`, `∇I_uv`; `J_uv`, `J_Z`; computed 4×3 `H_x`.
5. Signed residual `r=[a-I(u,v),Z-D(u,v)]`; per-channel `R` **including interpolation and model variance**; standardized depth residual.
6. Exact prior-predictive `P^-_xx`; full 4×4 innovation covariance `S=H_xP^-_xxH_x^T+R`, `nu`, NIS, threshold and accepted/rejected reason; **do not equate position sigma with innovation sigma**.
7. Pixel ownership, frontmost point ID, occlusion, duplicate suppression and **which actually admitted row(s)** entered the final 6×6 solve.
8. The identical row recomputed at **prediction and accepted posterior**, plus original canonical position, with changes in pixel support explicitly reported.

**Sign invariant from the public implementation:** `r_D=Z-D`, while `H_D=∂(D-Z)/∂x`. Derive the correct Kalman RHS with that convention. Verify actual local code instead of copying formulas blindly.

**Depth identity limitation:** a smooth 2×2 neighborhood establishes local interpolation continuity, NOT correspondence to the original material point. Projected depth consistency, even across multiple cameras, cannot by itself identify tangential position on a textureless plane.

## 6. Independently reproduce the actual 6D IEKF correction

For each decisive accepted update, reconstruct the **complete** local normal system, including all admitted cameras and RGB/D channels. Evaluate the actual prior-offset correction when linearizing away from `s^-`:

\[
 \Lambda^-=(P^-)^{-1},\quad
 A=\Lambda^-+H^TR^{-1}H,\quad
 b=H^TR^{-1}r-\Lambda^-(s_j-s^-),\quad
 \delta_j=A^{-1}b.
\]

*Use the sign convention in the actual implementation; the displayed algebra assumes `H=∂h/∂s`, `r=-h`.* If the algorithm uses a different accepted-step formula, reproduce that formula **exactly**. A real accepted result may be a previous iterate or prediction, not the final computed `s_j+delta_j`.

Check `A`, `b`, `delta`, Cholesky factor, predicted cost and accepted state against **independent NumPy float64** with justified tolerances. Compare depth/RGB gradient contributions at fixed full Hessian:

\[
 b=b_{prior}+b_D+b_{RGB},\qquad
 \delta=A^{-1}b.
\]

Report `A^{-1}b_D` and `A^{-1}b_RGB` (same `A`) *and* separate counterfactuals in which a channel is omitted and the **whole Hessian** is recomputed. These mean different things; do not conflate them. Ensure velocity coupling is explicit:

\[
 \Delta v=K_v r
\]

for a linearization at the prior, or its **correct iterated** analogue with the prior-offset term. **Do not** use single-row scalar Kalman formulas to explain a joint multi-camera update.

Verify the actual `P_{vx}` at the moment velocity first appears. Record whether a depth residual of only tens/hundreds of micrometers can produce a millimeter-scale 3D step and a nonzero velocity, and quantify it in the actual trace.

## 7. Specific depth hypotheses — distinguish initiator from amplifier

### H1: Interpolation/model discrepancy at physically correct stationary position

At canonical/static reference and (when available) actual GT surface position, evaluate the **same** four original depth pixels:

\[
 D_{metric}=\sum_jw_jD_j,\qquad
 D_{inv}=\left(\sum_j w_j/D_j\right)^{-1}.
\]

For one exact planar surface in a pinhole camera, inverse depth is affine, but do **not** assume all observed neighborhoods are planar or single-surface.

Compute signed depth residuals under each, 2×2 slope, subpixel fraction, source vs other camera, per-frame consistency, depth discontinuity and analytic gradient. Verify `D_inv` Jacobian and first-order variance with an independent oracle:

\[
 \nabla D_{inv}=-D_{inv}^{2}\sum_j \nabla w_j/D_j,
 \quad
 \mathrm{Var}(D_{inv})\simeq D_{inv}^{4}\sum_j(w_j^2\sigma_j^2/D_j^4).
\]

Then **offline only**, at identical position/prior/cameras/**frozen row identities**, compare the **re-solved complete 6D IEKF correction** with original metric depth versus counterfactual inverse depth. Report improvement in **the original accepted delta_v and delta_x**, not merely reduction of an isolated residual.

### H2: Wrong-surface or tangentially ambiguous depth association

At first accepted row after divergence, determine whether the four depth pixels belong to the same **physical surface/part** as the point. Use GT part/surface buffers **only offline if they actually exist**. If only labels of reconstructed points are available, do **not** claim exact material-surface identity: report `identity_unproven` with supporting geometry.

Analytically verify the planar tangent nullspace: depth consistency constrains a point to a surface, not its material identity along the surface. Check `H_D` rank/gradient direction, RGB texture information, signed depth and camera coverage.

### H3: Prediction covariance inflates row admission, then a precise depth row drives the update

Compute **actual** `S_DD=H_DP^-_{xx}H_D^T+R_D`, full RGBD NIS, and the configured `rgbd_chi2` vs `depth_front_only` rule. For every suspected wrong association, prove it *passed* the actual gate before calling the gate causal. Report actual √`R_D`, √`S_DD`, and position covariance separately.

Check the effective depth SD (the previous outer result reported ~55 μm, but the **standard F1 may use a different noise configuration**). Recompute all variance terms and units; do not import outer numbers into the standard run.

### H4: Position–velocity covariance turns biased position information into persistent motion

Reproduce `delta_v` and its camera/channel contributions using the **accepted F1** system. Then propagate that accepted state **with zero further measurement corrections in an offline reference** and compare with actual trajectory. Log first time drift grows despite zero accepted observations, and exact frame of eventual observation recovery.

### H5: Standard F1 nonlinear iteration / acceptance bug

Investigate **ONLY the standard** `last_consistent_iterate` (or recorded actual acceptance policy), including:

- Why a candidate is admitted or rejected; whether it actually **decreases a single fixed-support nonlinear posterior objective** (the standard acceptance may be a chi-square adequacy check instead—record, do not misrepresent).
- Whether a candidate crosses image-pixel boundaries or changes 2×2 support while a Jacobian or gate is evaluated at a different position.
- Whether state changes during the IEKF iterations but row ownership remains frozen; whether final `recheck_self_occlusion` is actually applied.
- Whether `P^+` is computed with information from rows that are no longer valid at **the accepted state**.
- Whether `P^+` shrinks without an accepted measurement, or if an unacceptable step can leak through a fallback state.
- Whether uncertainty growth occurs because valid observations are lost after a false early velocity, not because of an arithmetic covariance bug.

An accepted state with a low chi-square statistic is **not necessarily** a correct physical correspondence or a decrease in one globally fixed objective. Determine what the code promises and whether it actually satisfies that promise.

### H6: Moving-part point loss in standard F1

Using **the standard F1 saved original point IDs**, quantify per frame: has RGB-D rows, prediction-only, row lost by reason, offscreen, outside FG, wrong-surface evidence, 3D distance to GT-transformed part (when valid) and still present in output arrays. Compare a successful and failed moving point's **first divergence**, not just final position. Do not confuse points retained in memory with physically recovered moving surface.

### H7: Numerical bug or data/provenance error

Independently validate camera depth **Z vs Euclidean range**, `T_cw` convention, projection/backprojection, row Jacobian signs, correct `dt`, exact 6×6 covariance propagation, `P_vx`, SPD/Cholesky, source-measurement double counting, interpolation variance and point-ID ordering. Any discrepancy beyond quantified float precision is a concrete bug lead, not a reason to relax tolerances.

## 8. Controlled causal counterfactuals — offline, not new inference

Use a few **saved F1 prediction states**, exactly their actual fixed camera/row support and noise. Run the following **independent local algebraic counterfactuals only**:

| Counterfactual | What changes | What stays fixed | Question |
|---|---|---|---|
| C0 | Nothing | Exact recorded F1 equation | Can we reproduce the accepted correction? |
| C1 | Metric depth ↔ inverse depth | Same 2×2 cells, cameras, states, process and row IDs | Does interpolation bias *quantitatively* cause the false update? |
| C2 | Remove depth channel | All original states/RGB factors, correct re-solved Hessian | Would RGB alone inject velocity at the onset? |
| C3 | Exclude a **proven** wrong-surface depth row | Same prior and other rows, correct re-solved Hessian | Is that row necessary for the observed correction? |
| C4 | Remove a measured early velocity injection in a *pure prediction simulation* | Original `F,Q`, no later measurements | How much subsequent departure comes from that velocity? |
| C5 | Original row at initial prediction vs reprojected accepted state | Same raw frames and original PID | Does sampling/gating change in a way acceptance does not audit? |

Counterfactuals do not produce legitimate tracked trajectories and must never be substituted for production inference. GT-based row exclusion is an *oracle audit*, not an allowed estimator input. Record signed changes in the 6D correction, objective/cost where mathematically comparable, and whether row validity changes.

## 9. Independent minimal synthetic oracles (cheap tests, no additional scene)

Create independent deterministic **NumPy float64** tests, not another full SAPIEN inference run:

1. **Exact slanted plane:** ground-truth depth at fractional pixels; arithmetic vs inverse depth, signed interpolation bias, normal and tangential Jacobians; known stationary point.
2. **Depth discontinuity:** two separate planes within nearby pixels; show a locally valid or falsely admitted neighborhood can belong to the wrong surface; test actual same-surface policy without pretending that a local spread test establishes identity.
3. **6D Kalman step:** known `P^-` with nonzero `P_vx`; one depth measurement and then full RGB-D multi-camera rows; independent dense oracle against actual F1 GPU result.
4. **Prediction and loss:** introduce one deliberately erroneous accepted velocity, then remove later measurements and show exact displacement/covariance propagation; verify no artificial information gain.
5. **Acceptance:** synthetic candidate whose linearized chi-square is acceptable but actual nonlinear cost increases or changes support; establish what the original F1 does. This tests behavior, not a requirement to modify it.

Distinguish a **mathematical model limitation** (e.g. planar tangent ambiguity) from a **software bug** (e.g. wrong Jacobian sign). Do not introduce any new thresholds to make tests pass.

## 10. Required compact outputs

Write outside the original experiment root, for example:

```text
forensics/USB_F1_inner_drift_20261030/
    README.md                     # exact commands, provenance, one-scene scope
    root_cause_report.md          # main causal argument; confidence per claim
    selected_point_ids.csv
    f1_state_timeline.csv         # every frame for selected IDs
    f1_inner_iterations.jsonl     # only decisive original IEKF iterations
    depth_samples_first_events.jsonl
    f1_row_admission.csv
    f1_update_oracle.csv
    f1_channel_counterfactuals.csv
    f1_moving_point_survival.csv
    oracle_tests.txt
    hypotheses.json              # H1–H7 status + numerical evidence
```

Up to five labeled small image/depth crops if they are actually needed for identity diagnosis. No massive repeated `.npy` arrays, full frame dumps, new 100k/300k benchmarks, additional synthetic scenes or real Laptop solve.

For each H1–H7, classify `CONFIRMED`, `PARTIAL`, `DISPROVED`, or `UNPROVEN`; include code path and line numbers, numerical trace and supporting/rejecting observations. **One or more causes may be present.** Separate:

- **Initiator:** first wrong measurement or bug that generated the accepted error.
- **Amplifier:** the gain and velocity prediction that increase displacement.
- **Association failure:** why valid RGB-D rows are lost or why wrong ones become accepted.
- **Consequence:** why static points depart and moving-part point IDs apparently disappear.

## 11. STOP RULE — no outer-loop work

Do not propose or implement a corrected filter until the report can state for **at least three standard-F1 static failures**:

> At F1 frame **k**, inner iteration **j**, camera **c**, and source/current pixel **(u,v)**, the actual four depth samples yield signed **r_D** and exact **H_D**, **R_D**, **S_DD**, NIS and gate decision; the accepted 6D F1 normal solve produces **delta_x** and **delta_v**; these values agree with a separate float64 oracle; the next prediction integrates the injected velocity; the later observation trace shows exactly when the track loses or changes its surface evidence. The initiating discrepancy is [H1/H2/H3/H5/H7], the amplification is [H4], and these particular links are proven or remain unproven.

If source code or saved data cannot support a link, identify the missing trace and collect **only that trace from the standard IEKF**. No IEKS rerun, no full-dataset sweep, and no speculative tuning. Finish by listing at most **two surgical, evidence-backed candidate corrections** for user review, **not implemented**.
