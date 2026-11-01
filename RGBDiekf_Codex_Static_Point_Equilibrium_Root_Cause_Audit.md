# RGBDiekf — static-point equilibrium failure: diagnostic-only Codex investigation

## Mandate

**Investigate and report, do not fix.** Determine exactly why a point whose physical 3D position is stationary, whose frame-0 estimated velocity is zero, and whose identity is preserved by forward transport acquires an accepted nonzero position/velocity update and eventually escapes.

The question is **not** how to implement a new tracker, point association, constraint, mask loss, optimizer, motion prior, or correspondence search. The existing residual is a **direct sample of measured RGB-D at the transported 3D point's projection**; preserve that design. Identify the first causal event on **today's local working tree**, not an old report.

Success = a reproducible, numerical, frame-by-frame explanation of *which measured rows generated which state correction, why the IEKF accepted it, how much velocity followed from the prior covariance, and how the error persisted through visibility changes*. Explicitly distinguish measured facts from hypotheses.

## Non-negotiable constraints

1. **No modifications to inference.** Do not modify the 6D filter equations, update acceptance, prediction, gates, camera geometry, masks, reference appearance, process/measurement noise, dataset, or runtime defaults. No tuning, motion clipping, EKF replacement, velocity damping, optimization, surface correspondence, nearest-neighbor matching, rigid groups, QP, EP, multiview mesh/surface construction, or new mask-measurement inference.
2. Only add **opt-in trace instrumentation / offline diagnostic code and tests**. Default runs with tracing disabled must be bit-for-bit equivalent to the current baseline (and should not allocate extra large arrays).
3. Work against the **local** repo, likely `~/Joint3DGS/RGBDekf`; it has local uncommitted changes and may differ from GitHub. Inspect, do not overwrite or revert them. Do not commit or push without user authorization.
4. Inference uses **only existing transported states and measured RGB-D/foreground masks**. Ground truth may be read exclusively by an **offline evaluator/selector**, and must not be fed into any filter, gate, correction, or appearance sample.
5. Use real frame times from dataset metadata, never assume `dt=1/25` or `1/59` unless the scene's metadata confirms it.
6. Preserve point IDs and subsampling manifest. Never mix point IDs from distinct run/subsampling configurations. `rendered_mask_measurement.mode=off` means *new rendered-mask observation disabled*; the existing foreground masks remain required for RGB-D sampling.
7. Do not leave a heavy job running without a direct deliverable. Reuse prior baseline runs only if their code/config/dataset/subsample hashes match; otherwise execute the smallest reproducible baseline required.
8. Save **small, targeted** traces and numerical reports by default; no full per-point per-frame debug images or covariance dumps. A few selected pixel crops are acceptable if needed.

## 0. Snapshot the current experimental contract (before running)

Record in report:

- `git rev-parse HEAD`, `git status --short`, hashes of all touched files, Python/Torch/CUDA versions, GPU, run command, timestamp, source tree hash or clean equivalent, and whether all experimental files were modified relative to HEAD.
- Exact `train.py` CLI, policy, noise config, timing, camera intrinsics/extrinsics, `--pixel-reuse-assignment`, `--covariance-precision`, `max_iekf_iters`, update acceptance mode, `row_innovation_gate`, `final_visibility`, and mask mode.
- Physical units/values of initial `P_xx`, initial `P_vv` / `v99` (if declared), `P_xv`, process `Q(dt)`, RGB/depth `R`. **Label declared uncalibrated values as such.**
- Confirm whether these exact modules/functions are still in the local code: `scene/endpoint.py::endpoint_samples`, `estimator/free_projective_run.py::run_free`, `estimator/point_kalman_cuda.py::{predict, PointMeasurement,update_points}`, `estimator/projective_terms.py::build_terms`, `estimator/projective_sensor.py::sample`, and `estimator/projective_visibility.py::{visibility,apply_frozen_assignment}`. If moved/renamed, use actual implementation, not stale names.
- Show the exact selection of the canonical frame-0 RGB `rgb0[i]` (source camera and source pixel), position, and covariance for each traced point.

## 1. Reproduce baseline and choose *verified physically static* test points

**Primary analysis is forward-only, rendered-mask measurement OFF**. Do not include experimental four-pixel mask measurement in this diagnosis.

Datasets: `storage_45135`, `USB_100109`, `stapler_103111` if those are the current scene identifiers. Include the existing **completely static synthetic fixture**. If one real recorded scene is available and has confirmed static 3D references, repeat a small check there; otherwise label that test *not available*, do not fake it.

- Use current camera-center correction and current gates / acceptance. **Do not use old 29a23a7 trace output as evidence for this build.**
- Define *physically static* with scene ground-truth kinematics/part transforms in the offline evaluator or the known completely static fixture. Do **not** infer physical staticness from estimated zero velocity. For transformed GT points, preserve the canonical source surface point and transform it using the true part pose. If exact point identity ↔ GT part correspondence is unavailable, explain this and restrict to points with grounded static identity; never silently assign it.
- Offline selector should deterministically choose, per scene, e.g. up to 10 worst-final-error truly static points, up to 5 stable truly static controls with similar observed camera count/depth, and up to 5 additional truly static points with high initial innovation (if available). Those are **audit samples, not a tuning subset**. Output original source ID, canonical point ID, source camera/pixel, GT static classification evidence, selection rationale. Do not hardcode old IDs unless hashes match.
- For the first 3–5 frames and the earliest error onset for each selected point, rerun with tracing enabled. Continue just long enough to establish causality; if onset is late, trace a compact window of preceding frames and the onset frame. Include points with **no accepted RGB-D measurement** as controls.
- Report the baseline numbers before diagnosis: GT position error distributions for truly static points (median/p90/p99, >10 cm), velocity errors, fraction with no valid camera rows, acceptance status counts, and runtime. Also record numerical position drift from *the point's own canonical frame-0 coordinate*, separately from GT error (the backprojected canonical sample is not necessarily exact physical surface position).

### Event definitions (avoid arbitrary new gates)

Let `k0` be the anchor frame and `s^+_{i,k}` the published posterior. For each audited point, identify:

1. **First accepted nonzero correction:** earliest `k>k0` where `s^+_{i,k} != s^-_{i,k}` above numerical roundoff (log exact values; only ignore changes below a justified numerical epsilon derived from dtype/scale).
2. **First static-error increase:** earliest frame where `||x^+_{i,k}-x^{GT}_{i,k}|| > ||x^-_{i,k}-x^{GT}_{i,k}||` by more than numerical roundoff. This may differ from event 1; do not conflate them.
3. **First significant departure:** first frame crossing the *reporting* threshold `>0.10 m` GT error, if any. Show the entire causal chain leading up to it, not only the endpoint.
4. **First spurious velocity:** earliest accepted change in `v` and its relation to the first position correction; do not use a made-up speed cutoff.
5. **First loss/recovery of observations:** when valid RGB-D row count drops to zero and/or reappears, with reasons per camera.

## 2. Trace the exact data at each event, not an abstract “association”

For each selected point and relevant frame, capture the **predicted state and all actual direct projective RGB-D rows**:

**State and covariance**

- `x_prev_plus`, `v_prev_plus`, `P_prev_plus`; actual `dt`.
- `x_minus`, `v_minus`, `P_minus` (6x6), `P_xx`, `P_vx`, `P_vv`, eigenvalues/condition numbers; `F`, `Q(dt)` or their values/hashes.
- Ground-truth point `x_GT` only in posthoc report, and the position error at prediction and posterior.
- `x_plus`, `v_plus`, `P_plus`, acceptance code, accepted IEKF iteration, convergence code, number of valid rows, NIS and degrees of freedom, recorded posterior cost.

**Per camera at prediction and every chosen IEKF iterate**

- Predicted pixel coordinates `(u,v)`, camera optical `Z`, pixel interpolation base `(floor u, floor v)`, fractions `du,dv`, the **four source RGB values, four depths, four mask values**, and their `raw_valid` flags. Also nearest-pixel position used for sparse frontmost visibility (do not confuse nearest-pixel visibility with bilinear RGB-D sampling).
- Source/reference RGB `rgb0[i]` and source camera/pixel; measured bilinear RGB and depth.
- Visibility code before frozen assignment, frozen visibility code, actual row-admission code, pixel frontmost point's ID and `Z`, and actual measured depth. Record whether the camera's row was **used** (`reason == VALID`).
- Full residual `r=[rgb0 - I_t(u,v); Z-D_t(u,v)]` (4 elements), analytic `H_x` (4x3), `R_diag` (4), innovation covariance `S_r = H_x P^-_xx H_x^T + R`, prior-linearized NIS, chi-square threshold and gate outcome.
- Separate per-row/per-camera contributions: `g_RGB=H_RGB^T R_RGB^-1 r_RGB`, `g_D=H_D^T R_D^-1 r_D`, `I_RGB=H_RGB^T R_RGB^-1 H_RGB`, `I_D=H_D^T R_D^-1 H_D`. Sum across actually used cameras. Do not label them “matches”; these are direct image samples.
- Specifically test frontmost invariant: if an actually nearer **observed** surface is statistically in front of the predicted point, check whether depth row is rejected; if another transported point wins projected frontmost ownership, the hidden point must not use that camera's RGB-D row. Report any violation with concrete frame/point/camera/pixel. **Do not assume this invariant is broken.**

**At canonical frame 0 / no-motion control**

- For a known static point held at its canonical coordinates, evaluate the *measurement function only* against actual frame-0 multiview images, and next-frame images. Do not perform a GT-aided update.
- Compare `rgb0` with the valid source-camera sample and with each other camera at the same instant. Is `rgb0` a view-dependent outlier? Are residual directions systematically aligned rather than random?
- Compare predicted optical depth and measured depth at the same actual point projection for each valid camera. Any nonzero reference-frame or static-frame measurement information gradient is evidence of a residual/model inconsistency (not by itself proof of a code defect; sensor noise/interpolation can create small nonzero residuals).

## 3. Reproduce the IEKF algebra and isolate where an error is created

The current projective model is an **implicit observation**, with the *existing* sign convention:

```
h_c(x) = [ I_{c,k}(pi_c(x)) - rgb0[i],
           D_{c,k}(pi_c(x)) - Z_c(x) ]
r_c(x) = 0 - h_c(x)
       = [ rgb0[i] - I_{c,k}(pi_c(x)), Z_c(x) - D_{c,k}(pi_c(x)) ]
H_c = dh_c/dx = [ dI/duv * J_pi ; dD/duv * J_pi - J_Z ]
```

For the fixed prior `(s_minus,P_minus)` at IEKF iterate `s_j`:

```
Lambda = inv(P_minus)
A_j = Lambda + sum_used H_6^T inv(R) H_6
b_j = -Lambda (s_j - s_minus) + sum_used H_6^T inv(R) r_j
delta_j = solve(A_j, b_j)
s_j+1 = s_j + delta_j
```

`H_6 = [H_x  0_(4x3)]`. The exact prior stays fixed for **all** iterations.

Instrument **every iteration** for audited points: `s_j`, used camera rows, `g_c`, `I_c`, `A_j`, `b_j`, `delta_j`, step score, cost `J_j`, camera row-set changes and pixel-cell crossings, acceptance flag, chosen published iterate. Confirm independent float64 NumPy recomputation at the actual traced inputs to numerical precision; do not call that proof of model correctness.

### Distinguish 3 failure modes using explicit numerical tests

**A. Observation gradient at a known static location**

- At `x=x_GT` (or canonical coordinate for the static fixture) and `v=0` evaluate the image-only information gradient `g=sum H^T R^-1 r`, **without applying an update**.
- Report `||g||` and its RGB vs depth and camera-wise contributions. If `g` is nonzero, trace which observed pixel intensities and gradients cause it. Do not label ordinary sensor noise a bug; look for persistent signed bias across time/cameras.
- Repeat at `x=x_minus` from the actual run. The two may be different; that difference is informative.

**B. Invalid nonlinear step / pixel-cell crossing**

For each proposed `delta_j`, calculate for every admitted camera:

```
Delta_u_linear = J_pi(x_j) * delta_x
Delta_u_actual = pi(x_j + delta_x) - pi(x_j)
h_linear_change = H_x(x_j) * delta_x
h_actual_change = h(x_j+delta_x) - h(x_j)
linearization_error = h_actual_change - h_linear_change
```

- Record pixel displacement, whether interpolation cell changes (`floor(u),floor(v)`), whether 2x2 mask/depth support changes, whether the measured surface/depth continuity test changes, and `||linearization_error||` **in R-whitened units** (plus per RGB/depth component). Compare actual vs predicted improvement in the *same* admitted residual rows. Distinguish an invalid cell-crossing from large in-cell nonlinearity.
- For diagnostics only, evaluate step fractions `alpha={1,1/2,1/4,1/8}` without publishing alternative states. Identify whether the full proposed step is outside the trusted local image model and whether smaller fractions better predict the measured residual change. This is **not** a request to implement line search.
- Test analytic `H_x` vs local central finite differences at nonboundary projections, choosing epsilon small enough to remain in the same bilinear cell. At pixel cell boundaries report nondifferentiability, not an arbitrary “failed Jacobian”.

**C. Acceptance can retain a worsening state**

- Record for each iterate: `J_j = (s_j-s_minus)^T Lambda (s_j-s_minus) + sum_actual_used r_j^T R^-1 r_j`, current `chi2_4m` threshold, and pass/fail.
- Crucially, test whether **accepted** `J(s_plus) > J(s_minus)` or `J(s_plus) > J(s_prev_candidate)`; label the actual rows in each cost. Existing `chi2` consistency acceptance is **not** the same as monotonic objective improvement.
- If a row becomes invalid or is dropped between evaluations, recompute a *fixed-row* objective over the **common admissible camera rows** as a separate diagnostic; do not falsely claim cost improvement from a changed number of rows. Report rows that were lost, gained, and rejected because of the proposed step.
- If the current code already prevents the old failure, mark **NOT REPRODUCED**, with example and counts. Do not import old root-cause assumptions into today's result.

## 4. Quantify precisely how a position correction becomes velocity

For any update using only position-dependent measurements, predicted covariance is partitioned as

```
P_minus = [[P_xx, P_xv],
           [P_vx, P_vv]].
L = P_vx @ inv(P_xx).     # 3 x 3, physical units 1/seconds
```

For a fixed-prior Gaussian update initialized at `s_minus` and solved consistently,

```
Delta_v_expected = L @ Delta_x_published
Delta_x_published = x_plus - x_minus
Delta_v_actual    = v_plus - v_minus
```

Compare `Delta_v_expected` vs `Delta_v_actual` per point/frame (relative/absolute error), not just an informal “0.5–1.0/dt” heuristic. Also report:

- Singular values / eigenvalues of `L` and `dt*L`, direction cosines between `Delta_x` and `Delta_v`, and contribution of `P_vv,0`, `P_xx,0` and `Q(dt)` to `P_vx` where algebraically applicable.
- Relative sizes of predicted positional uncertainty from prior position and velocity (`P_xx,previous`, `dt P_xv`, `dt^2 P_vv`, `Q_xx`). Show why `dt*L` approaches identity for specific points, or why it doesn't.
- Absolute `Delta_x` (mm), `Delta_v` (m/s), prior `v_minus`, posterior `v_plus`, **next prediction-only displacement** `dt_next*v_plus` and resulting GT error increase.
- The *first* accepted event that creates nonzero velocity, **not only the frame with maximum final speed**.
- A purely offline counterfactual `x_next_if_v_not_injected = x_plus + dt_next*v_minus` vs `x_next_actual_prediction = x_plus + dt_next*v_plus`. This isolates the contribution of newly injected velocity without modifying the filter.
- **Exact trajectory attribution (required):** because the mean dynamics are `x_k^- = x_(k-1)^+ + dt_k v_(k-1)^+`, `v_k^- = v_(k-1)^+` and canonical initial `v_0=0`, with published update increments `delta_x_k = x_k^+ - x_k^-` and `delta_v_k = v_k^+ - v_k^-`, the final displacement decomposes *exactly* as

  ```
  x_T^+ - x_0^+ = sum_{k=1..T} delta_x_k
                  + sum_{j=1..T-1} (t_T - t_j) * delta_v_j
  ```

  Verify numerical closure of this identity from the saved trajectory. Report the **vector** contributions of accepted position corrections and propagated velocity injections, and isolate the contribution of the **first** spurious velocity event: `(t_T-t_j) delta_v_j`. For a scalar fraction, project components onto the net drift direction and state cancellation explicitly; **do not** divide magnitudes of components and pretend they sum to 100%. This establishes *how much* final escape is caused by persistent speed versus repeated new position corrections, without changing inference.

If the equality `Delta_v ≈ L Delta_x` does not hold, investigate whether velocity is being corrected by a second mechanism, whether different accepted iterates were mixed, or whether the test uses the wrong covariance; report the discrepancy as a potential implementation defect.

## 5. Prove the temporal/occlusion story rather than assuming it

For each drifting point, show a compact timeline (at least `k-2...k+3` around onset and around first >10cm departure):

| frame/time | x_GT | x_minus | x_plus | v_minus | v_plus | accepted cameras | reason by camera | first-order RGB/depth information gradient | accepted iterate | GT error | mask membership (diagnostic only) |

- Explicitly distinguish `prediction-only` frames (`nrows=0` or rejected all rows) from frames with accepted RGB-D corrections.
- On prediction-only frames verify `s_plus == s_minus`, `P_plus == P_minus` (subject to numerical precision), while **prediction** still advances `x` by `dt*v`. This demonstrates ballistic drift without falsely implying a hidden point was updated.
- Test at the onset frame whether the point was actually in front of the **real depth surface** vs behind it, and whether predicted-cloud frontmost ownership agrees. Include all four bilinear source pixels and their masks; do not label valid all-white foreground as background.
- Inspect whether a camera's row is frozen from a prior visibility decision despite the point moving into a different cell at a later IEKF iterate, and whether final `recheck_self_occlusion` prevents accepting a hidden-point row. Report a concrete counterexample or confirm invariant holds.
- When a point reappears, show whether its measured innovations now pull it back, are gated away, or remain ambiguous. Distinguish inability to correct because **no valid rows** from inability to correct because **nonzero rows are rejected**.
- Check the **mask-membership** of all projected points for diagnostics only. Do not run the experimental rendered-mask observation (four-pixel splats) and do not use masks to correct the point.

## 6. Targeted counterfactuals: analysis only, never change production policy

On the **first offending frame**, with the same fixed `(s_minus, P_minus)` and same measured RGB-D images, recompute a *single linearized* update **offline** under these row subsets, not as new rollout policies:

1. All actually accepted rows (exact current behavior).
2. RGB-only rows (all admitted cameras, same reference color).
3. Depth-only rows (all admitted cameras).
4. Leave-one-camera-out, each used camera in turn.
5. Source camera only vs all other valid cameras, when meaningful.
6. Prediction only (no measurements).

For each, report `Delta_x`, `Delta_v`, `J` contribution, predicted GT-error change (offline only), and whether a specific row dominates the normal-equation right-hand side. **Do not infer that disabling a camera is a recommended solution**. Correlated observations or changed nonlinear trajectory are not represented in this single-step diagnostic; state that limitation.

Optional *measurement-only* tests if ambiguous: at fixed `x`/`v`, compare actual frame-0 and next-frame image residuals for a known static point; compare nearest pixel vs bilinear **as diagnostics**; compare RGB reference against other valid cameras. No inference changes.

## 7. Verification tests (required)

Add narrow tests or independent scripts (opt-in, no heavy outputs):

1. **Exact static zero-residual oracle:** synthetic constant RGB-D image model for which a verified static point has `r=0`, correct masks, and zero velocity; `predict` preserves mean, `update_points` preserves mean, no mask-induced updates. Any failure = code/math defect.
2. **No-observation oracle:** all camera RGB-D rows invalid/occluded; no posterior correction at that frame, covariance posterior equals prior; state may move only through a previously estimated `v` during prediction.
3. **One-camera vs multi-camera occlusion oracle:** foreground moving occluder projects before a hidden static point; only the *occluded camera* excludes the hidden point's RGB-D row; another valid camera may still update it.
4. **Kalman cross-covariance identity:** test `Delta_v = P_vx inv(P_xx) Delta_x` for a single position-only linear-Gaussian update and for accepted IEKF states (tolerance justified by dtype). Check frame timing in `F,Q`.
5. **Image linearization oracle:** analytic Jacobians vs finite differences *inside the same bilinear cell*; deliberately crossing boundaries reports model error rather than a false analytic-Jacobian failure.
6. **Acceptance audit:** recorded `chi2` pass/fail matches local code, and any accepted candidate that worsens the prediction on identical rows is highlighted (not automatically classified as a violation of the *existing* acceptance policy).
7. **Baseline parity:** tracing disabled reproduces current uninstrumented baseline; synthetic and representative real/benchmark fixture results unchanged.

## 8. Deliverables, report shape, and decision criteria

Write `reports/static_point_equilibrium_root_cause_CURRENT.md` (date and local code fingerprint in title/header), plus a small machine-readable trace `reports/.../static_onset_trace.jsonl` or `.npz`, and a reproducible diagnostic runner e.g. `tools/audit_static_equilibrium.py` using *actual* repo interfaces. Keep large results in the existing designated experiment output root, not in Git. Reuse existing traces if correctly fingerprinted. No HTML dashboard or saved full images by default.

### Required report sections

1. **Executive verdict** (<=10 lines): which failure mechanism(s) were demonstrated **with current code**; which were not reproduced; what remains unknown. No proposed code patch in the verdict.
2. **Reproduction manifest:** exact command, commit/worktree hashes, input and configuration checksums, selected scene/IDs/source pixels/GT static basis, frame timing.
3. **Static-point onset table:** per scene/point: first correction frame, first GT-error increase, first accepted velocity, first >10cm error, camera rows, dominant RGB/depth/camera gradient, full-step pixel displacement, acceptance decision, max `dt*L` singular value.
4. **Detailed worked example** for at least one runaway static point per scene; show *actual numbers* for `r,H,R,g,I,delta,Delta_v,P_xx,P_vx`, accepted iterate, true vs predicted image changes, prior/candidate objective on comparable rows, time to escape.
5. **Matched stable controls:** at least two comparable truly static points that do *not* drift; explain the numerical distinction, rather than reporting only failures.
6. **Temporal visibility timeline:** demonstrate or refute actual hidden-point update and prediction-only ballistic propagation.
7. **Current-code mechanism ranking** (supported evidence only):
   - E1: nonzero residual/information gradient at truly stationary geometry (RGB vs depth, by camera).
   - E2: nonlinear image linearization invalid due to cross-cell step / surface support change.
   - E3: accepted iterate has worse actual comparable-row objective than prediction.
   - E4: correct Kalman `P_vx` turns a positional correction into high velocity.
   - E5: long prediction-only or rejected-observation interval permits persistence.
   - E6: a real violation of projected/masked visibility or foreground sampling invariants.
   Mark each **CONFIRMED**, **NOT REPRODUCED**, or **UNDETERMINED** with count and cited trace IDs. Do not assert “no coding error” merely because NumPy reproduces the formulas.
8. **Minimal next diagnostic or correction target**, *not implementation*: the single most supported issue, exact file/function and corresponding observed evidence. Any suggested fix must be explicitly separated from findings.
9. **Regression test results** and instrumentation overhead / data volume.

### Strict reporting rules

- Do not claim a static point should produce exactly zero individual residual under noisy images; distinguish nonzero random noise from a systematic bias or invalid step.
- Do not claim a background pixel was sampled unless actual source `mask[u0:u0+2,v0:v0+2]` proves this for the exact code version; foreground validation currently requires all four mask pixels white.
- Do not describe projection-based residual sampling as nearest-neighbor point association or as matching another point.
- Do not use data from a historical trace to claim today's policy still fails. Label old trace results separately as context only.
- Do not silently alter `v99`, `Q`, `R`, sample gradients, gates, masks, optimizer, or acceptance while executing this audit.
- If a necessary GT-to-point mapping, scene, command, or artifact is unavailable, state the **specific blocker** and complete the synthetic and currently feasible portions. Do not invent missing values.

## Completion criterion

The investigation is complete only when it answers with evidence:

> **For a physically static point, what is the earliest nonzero accepted image-driven correction? Which exact measured RGB-D pixel samples generated it, what does the local Jacobian predict versus what the image actually does, why did the IEKF accept the result, and what numerical fraction of its later drift is attributable to the resulting injected velocity rather than subsequent new updates?**

Stop after delivering this audited explanation. **Do not implement a solution or launch parameter searches.**
