# Codex implementation contract — Depth-Sorted Temporal RGB-D IEKF

**Date:** 2026-10-30  
**Repository:** `https://github.com/SamueleMara/RGBDiekf`  
**Local repository used in previous runs:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Scope:** F1/B1 free-point filters first. Preserve every existing baseline and downstream stage.  
**Primary mode:** `full` (fixed-lag joint state/observation inference with shared-reference uncertainty).  
**Ablation:** `simplified` (same observation model but fixed reference state and approximate marginal independent covariance).  
**Control:** `legacy` (current production projective-RGBD IEKF, unchanged).  
**Not authorized:** change prior process noise, hide/move synthetic GT into inference, assign static/dynamic labels, add arbitrary speed/displacement thresholds, revert to Gaussians/surfels/normals/meshes, or run all 60 frames on every dataset before tests pass.

## 0. Codex workflow and non-negotiable acceptance criteria

1. Inspect the current working tree **before editing**, including any uncommitted fifth/sixth-pass forensic tools. Record `git status --short`, commit SHA, `python train.py --help` and the test command. Do not reset or overwrite uncommitted changes.
2. Implement on a new branch or separate module set, leaving `--observation-model legacy` the exact existing path. Existing baseline default must remain `legacy` unless the user explicitly approves promotion.
3. Implement the **same z-buffer, camera-specific reference lifecycle, temporal warp, hard-mask validity and Jacobian** for `full` and `simplified`. Only the uncertainty treatment and state elimination differ.
4. **Full mode is not allowed to mean** “use `R_current + R_reference` while discarding correlations.” It must build a joint Gaussian/least-squares system over temporally linked point states and *unique latent clean reference/current RGB-D pixels*, process each raw RGB-D observation **once**, and retain cross-time covariance after marginalization. If the chosen implementation uses an algebraically equivalent correlated-noise formulation, supply a numerical equivalence proof against the explicit latent-variable oracle.
5. A point not selected as frontmost in camera `c` receives **zero direct update** from that camera. With no selected valid pixels in any camera, and under an independent-point prediction prior, it remains prediction-only with `s+ = s-`, `P+ = P-`. Do not confuse that with constant world position: nonzero velocity still propagates.
6. No mask=0 RGB/depth value contributes to any factor. No invalid or missing depth contributes. Unexplained foreground and visible predicted points outside the observed mask must be reported explicitly; do not improve an objective by dropping rows.
7. Verify direct finite differences for the **full nonlinear observation**, CPU double-precision dense oracle versus GPU, JVP/VJP adjoint identity, exact covariance/Schur equivalence, and no source measurement double-use.
8. Implement tests first on minimal synthetic fixtures, then 3–11-frame traces of actual SAPIEN scenes; expose reproducible commands but **do not launch long full-scene production runs** autonomously. The operator will run the commands in §17.
9. Do not claim exact global Bayesian inference. `full` means joint fixed-lag local Gaussian inference *conditional on hard visibility and the local temporal warp*. Depth-order/association and local linearization may be ambiguous; track these approximation limitations in the report.

## 1. Source-code map: inspect precisely these entry points

Paths below are verified on the GitHub default branch as of this specification, but **line numbers may change in the user's dirty/local tree**. Search by symbol when they differ.

| File / symbol | Existing behavior | Required integration |
|---|---|---|
| `train.py` `parse` (~52–136), `run_scene` (~172–300), `STAGE_SOURCES` (~36–48) | CLI, hash/resume metadata; F1/B1 runner; legacy default `--pixel-reuse-assignment per-frame` | Add `--observation-model {legacy,temporal-rgbd}`, `--temporal-mode {full,simplified}`, `--temporal-lag-frames`, `--temporal-reference-policy {latest-valid}`, `--temporal-warp-noise {measured,config}` and `--temporal-warp-noise-config`; record versions/hashes and mode in manifests. No hidden fallback. |
| `scene/endpoint.py` `backproject`, `samples`, `endpoint_samples` (~48–73) | Source camera/pixel, RGB, XYZ and source position covariance | Retain source provenance and initialize a source-camera valid reference at frame 0 (or frame T for B1). In **full**, avoid counting the source RGB-D depth once as an independent 3D position prior **and again** as an independent raw image factor. |
| `scene/dataset.py` `Dataset` (~8–36) | Calibrated source images, timestamps, frame iteration | Preserve original frames/camera indices and time signs; one image/frame loaded per camera. |
| `scene/dataloader.py` | Native SAPIEN RGB-D/masks, camera definitions | Do not change parsing/calibration conventions. |
| `scene/noise_calibration.py`, `scene/config.py` | Sensor noise and process priors | Keep physically declared `Q`; derive RGB-D observation noise from existing calibration, and separately document temporal warp/model-mismatch noise with provenance. |
| `estimator/projective_camera.py` `CameraRig` | `project`, `project_rows`, `J_pi`, `J_Z` | Reuse exactly, including camera conventions. |
| `estimator/projective_sensor.py` `MeasuredFrame` (~54–75), `sample` (~89–129), `sample_cpu_oracle` (~135) | Bilinear RGB-D; gradients; strict four-neighbor validity and same-surface checks | Reuse raw measured fields and analytical bilinear derivatives. Add an indexing API returning **raw pixel keys and interpolation weights** so full mode can track shared sensor noise without duplicating observations. |
| `estimator/projective_visibility.py` `visibility` (~20–56), `apply_frozen_assignment` (~67–81) | Round-to-pixel z-sort, one winner, excludes duplicate others; assignment optionally frozen for whole frame | **Do not reuse its row-admission policy unchanged as a renderer.** Add a `first_hit_pixel_owner` generating explicit `(camera,frame,pixel)->point`, projected depth, count/conflict and tie provenance. Freeze only **within each linear solve**; recompute after accepted nonlinear updates. |
| `estimator/projective_terms.py` `build_terms` (~62–96), `h_value` (~99–104) | Legacy `r=[a-I_t(pi(x));Z-D_t(pi(x))]`; gradients of current image | Keep for `legacy` control. Add separate temporal predicted-measurement operator, not flags hidden inside old terms. |
| `estimator/point_kalman_cuda.py` `transition` (~62), `process_noise` (~67), `predict` (~76), `PointMeasurement` (~164–268), `update_points` (~270–368) | Fixed-prior independent 6D IEKF; current row gates/acceptance, numerical traces | Reuse prediction; preserve `update_points` unchanged for legacy. Implement separate `update_points_temporal_simplified` and `solve_temporal_full_lag`. Do not call legacy `PointMeasurement` in temporal modes. |
| `estimator/free_projective_run.py` `build_catalogue` (~37–47), `run_free` (~50–173), `_store` (~246), `_frame_numeric` (later) | Builds F1/B1 catalogue, predicts, updates, writes arrays and numeric traces | Insert a temporal runner branch; create/refresh per-camera reference state **after accepted updates**; write new reference, visibility, solver and covariance diagnostics. Preserve original output schema. |
| `estimator/rendered_mask_measurement.py`, `estimator/rendered_mask_iekf.py` | Independent legacy rendered-white-point mask option | Default **off** for temporal mode; do not combine old separate mask filter until a common objective/covariance is explicitly derived. Continue to use observed masks as hard validity/feasibility constraints. |
| `debug/projective_debug.py`, `tools/evaluate_projective_gt.py`, `tools/root_cause_trace.py` | Debug, post-hoc GT, forensic traces | Extend numeric reports, add reference/first-hit visualizations, evaluate GT only in a post-hoc process. |
| `docs/PIPELINE_CODE_MAP.md`, `README.md` | Existing code architecture, flags, tests | Update with new modules, limitations, commands and return codes. |

**Relevant RigEuler reference, for algorithms rather than copying Gaussian geometry:** `SamueleMara/RigEuler3DGS/estimator/renderer_bridge.py` (`build_linearization`, `apply_H`, `apply_HT`) and `submodules/diff-gaussian-rasterization/cuda_rasterizer/iekf.cu` (freeze contributor list during linear solve; rerender and resort after accepted steps). We use only the contributor-cache lifecycle, **not** alpha/opacity, conics, Gaussian scales or full splatting.

## 2. Exact geometrical and observation definitions

For point `i`, global state `s_{i,k}=[x_{i,k};v_{i,k}]` in R^6, covariance `P`, and fixed calibrated camera `c`:

```
F(dt) = [[I3, dt I3], [0, I3]]
s_{i,k}^- = F(dt) s_{i,k-1}^+
P_{i,k}^- = F P_{i,k-1}^+ F^T + Q(dt)
u_{i,c,k} = pi_c(x_{i,k})        # image coordinates
z_{i,c,k} = Z_c(x_{i,k})         # camera depth
```

Use actual dataset timestamps, including negative time increment for B1. Do not mix frame indices and acquisition times.

### First-hit renderer

For every camera `c`, frame `k`, pixel `p`, choose:

```
kappa_{c,k}(p) = argmin_{i in candidates covering p} z_{i,c,k}
```

Every pixel has **zero or one** winner. For the minimal version `candidates covering p` uses the existing declared nearest-integer-pixel projection (do not invent Gaussian footprints). Explicitly log the chosen pixel-coordinate convention, tie rule (deterministic point ID), camera, point and depth. No second winner contributes RGB-D information to that pixel. Winning point `i` in cameras A and B may legitimately receive rows in both, provided each camera has a valid local temporal reference. Occluding points need not have reference histories to block other points.

At the anchor frame the source camera/pixel relationship is **known by construction**. Distinguish the source identity map from the unrestricted all-camera depth render; do not silently overwrite the known source map with cross-camera duplicates at frame 0. Full and simplified must use an identical documented rule for initialization and subsequent visibility. This is important for points Storage 8444 and 45588.

### Camera-specific temporal reference

Each `(i,c)` has a reference `Ref(i,c)`, or `UNINITIALIZED`:

```
ref_frame tau, ref_time_seconds, ref_state_id
ref_global_point_id
ref_projected_uv, ref_camera_depth
ref_owned_pixel_keys / local interpolation support
ref_valid_foreground_and_depth
ref_raw_camera_pixel_keys + exact interpolation weights
ref_state_covariance_link (full mode)
ref_reason / number of frames since last valid observation
```

At time `t`, a currently selected and reference-valid point uses:

```
delta_u = pi_c(x_{i,t}) - pi_c(x_{i,tau})
delta_z = Z_c(x_{i,t}) - Z_c(x_{i,tau})
q(p) = p - delta_u
pred_rgb(p) = I_clean_{c,tau}(q(p))
pred_depth(p) = D_clean_{c,tau}(q(p)) + delta_z
r(p) = [I_measured_{c,t}(p) - pred_rgb(p),
        D_measured_{c,t}(p) - pred_depth(p)]
```

`I_clean,D_clean` are **latent noise-free source images in full mode**, inferred from the raw camera observations; they are replaced by the raw measured reference interpolant in simplified mode. RGB and depth are always referenced in the **same camera**, so no direct source-camera RGB comparison across different cameras is required.

Reference patches are **not** assumed rigid in 3D. This is a locally translating 2D RGB-D pattern (plus camera-depth translation). Treat it as an explicit local approximation; do not implicitly invent normals or surface tangents. Reject invalid interpolation supports (out of image, masked-out, nonfinite depth, depth discontinuity under existing same-surface policy). A 2x2 reference support generally contains four *different* point IDs in a dense cloud; do not require all four IDs to equal `i`. Instead check raw foreground validity, depth continuity and whether the reference neighborhood is plausibly on the same visible surface rather than an occlusion boundary. Track support provenance.

### Analytic Jacobians, full two-time state

Let `gI = grad I_clean_{c,tau}(q)` (3x2), `gD = grad D_clean_{c,tau}(q)` (1x2), `J_pi,t`, `J_pi,tau` (2x3), `J_Z,t`, `J_Z,tau` (1x3), and let `h=[pred_rgb; pred_depth]`.

```
H_t^I   = - gI J_pi,t
H_tau^I = + gI J_pi,tau
H_t^D   =   J_Z,t   - gD J_pi,t
H_tau^D = - J_Z,tau + gD J_pi,tau
H_t     = [H_t^I; H_t^D] concatenated with zero direct velocity columns
H_tau   = [H_tau^I; H_tau^D] concatenated with zero direct velocity columns
```

`r=y-h`, so `d(r)/ds = -H`. The interpolation weights and gradients are differentiated at the same 4 raw pixel values used by the forward warp. For `h` vs clean reference raw-pixel latents `ell`, `H_ell` is their bilinear interpolation matrix. If the implementation freezes `gI,gD` and support for one linear system, refresh at next accepted relinearization. No differentiation through discrete sort switches.

**Critical consistency tests:** numerical forward difference of `h(s_t,s_tau,ell)` w.r.t. *both* states and all active reference clean RGB-D latent variables, plus JVP/VJP dot-product identity. At a support or depth-switch discontinuity, report `NONSMOOTH`, not a false derivative failure.

## 3. Full mode: correct latent-observation, fixed-lag probabilistic model

### 3.1 Why simple `R_t + R_tau` is not the full model

When `I_{c,tau}` is reused in several future residuals, its measurement noise is shared. Additionally, `x_{i,tau}` was itself estimated from images. An ordinary 6x6 IEKF treating each temporal residual as a fresh independent measurement overcounts evidence and loses `Cov(x_t,x_tau)`. **Do not call that full mode.**

### 3.2 Full graphical model (primary required implementation)

Within an active lag window `W_t={t-L+1,...,t}`, introduce:

- Point states `S={s_{i,k}}` at each retained time; keep previous window's **marginal prior information**, including cross-time/cross-point covariances induced by shared evidence.
- For every **unique** raw foreground pixel `(camera c, frame k, integer pixel p)` touched by temporal reference/current factors, one latent noise-free `ell_{c,k,p}=[RGB,D]` (4-vector). Duplicated uses of the same raw pixel reference the **same latent variable**, not new copies. In practice include only touched pixel keys, not dense images.
- One independent measured-image factor per unique raw pixel `y_{c,k,p} = ell_{c,k,p} + epsilon_{c,k,p}`, `epsilon~N(0,R_{c,k,p})` with declared calibrated sensor noise (and any justified intra-pixel RGB/depth covariance). **One raw pixel -> one sensor factor.**
- One temporal warp model factor for each selected current first-hit point-pixel association:

```
g_{i,c,k,p} = ell_{c,k,p}
              - sum_{a=1..4} w_a(q) ell_{c,tau,p_a}
              - [0,0,0, Z_c(x_{i,k})-Z_c(x_{i,tau})]
           = model_error ~ N(0, R_warp_{i,c,k,p}).
```

`w_a(q)` are exact bilinear weights of `q=p-pi(x_k)+pi(x_tau)`. `R_warp` represents deviations from the local brightness/depth-translation approximation, **distinct from raw sensor noise**. Its choice must be documented with provenance, units, calibration or explicitly declared assumption. No GT-based tuning or arbitrary hidden defaults. `R_warp` must be SPD when used as a Gaussian residual covariance. Calibration from image data is empirical Bayes and must be identified as such; do not assert fully calibrated uncertainty without validation.

- Markov process factors `s_{i,k}-F_{k,k-1}s_{i,k-1} ~ N(0,Q(dt))` and the prior from the previous marginalized window.
- At anchor time, source 3D position comes from the **same raw source depth** that is part of `ell`. Avoid independent duplicate depth information: implement a **single coupled source initialization factor** linking `x_{i,anchor}` to `pi^{-1}_{c_i}(p_i, ell_{c_i,anchor,p_i}.depth)` with calibrated pixel/depth/calibration uncertainty, plus an independent velocity prior; OR an algebraically equivalent joint anchor covariance that includes `Cov(x_anchor, ell_anchor)` and proves equality to this factor. Do not feed both the existing endpoint depth-derived `Pxx` as independent prior **and** the same raw depth observation again as independent factor. When working with pixel-integer coordinate uncertainty, record the convention used for the backprojection covariance.

**Key advantage:** observed RGB-D values enter the entire fixed-lag graph once, so reuse in time or by multiple candidate reference patches retains the proper correlations. Sensor noise is attached to raw camera pixels, *not* copied into independent time-difference rows. No `R_t+R_tau` shortcut is necessary in full mode. RGB-D observations are still masks-only.

### 3.3 Full MAP / information-form IEKF objective

Let `a` collect all retained point states and unique clean-pixel latents. Define

```
E(a) = 1/2 ||a_anchor - mu_anchor||^2_{Lambda_anchor}
     + 1/2 sum_process ||s_k - F s_prev||^2_{Q^-1}
     + 1/2 sum_unique_raw_pixels ||y_kcp - ell_kcp||^2_{R_kcp^-1}
     + 1/2 sum_temporal_factors ||g_{i,c,k,p}||^2_{R_warp^-1}
     + E_marginalized_past(a_boundary).
```

`E_marginalized_past` is the full Gaussian Schur-complement prior transferred from the previous lag window, not a separately reset covariance for each point. In the implementation the first anchor term is represented by the coupled source initialization above, not an extra duplicate data term. For nonlinear local warp, use a fixed-prior iterated Gauss–Newton/IEKF linearization, with an accepted-step strategy based on the *same* raw observations/factors and exact candidate rerender/warp evaluation.

At each iterate:

```
Lambda = Lambda_prior + J^T W J
b      = -grad E(a_j)
delta  = solve(Lambda, b)
a_trial = a_j + delta  # or damped/backtracked correction
```

Use float64 for state/covariance numerical oracle. Sparse CUDA/torch matrix-free `Jv` and `J^Tz` for production; never instantiate a dense `6N L x 6N L` matrix. Distinguish **MAP objective** from chi-square acceptance; monotonic decrease alone does not establish statistical adequacy, and adequacy alone does not guarantee convergence. Record both.

### 3.4 Correct covariance and reference-state marginalization

Partition active variables at frame `t` into published current point state `u` and nuisance/old states plus latent pixels `v`. Around the accepted linearization:

```
Lambda = [[Lambda_uu, Lambda_uv],
          [Lambda_vu, Lambda_vv]]
Lambda_u_marg = Lambda_uu - Lambda_uv (Lambda_vv^-1 Lambda_vu)
P_u_marg = inverse(Lambda_u_marg)
```

Use sparse Schur elimination or an algebraically equivalent symmetric solve. **Do not** replace this by a diagonal covariance or `inverse(Lambda_uu)` in full mode. `P_u_marg` is the reported current-state marginal; record cross-point covariance where shared pixel/reference factors create coupling. At window slide, eliminate only expired variables and preserve the *complete marginalized information* on the retained boundary. Repeated frame transitions must not implicitly reset the reference correlation.

The actual full cross-covariance blocks `Cov(s_{i,t},s_{i,tau})` and `Cov(s_{i,t},ell_{c,tau,p})` must be extractable for traced points for tests/debug. If an implementation compresses them in a low-rank/sparse approximation, label that implementation **approximate** and do not present it as exact `full`.

**Finite lag caveat:** exact local elimination for a given finite lag is possible, but nonlinear relinearization of previously marginalized factors cannot in general be recovered without their history. Record prior linearization/marginalization point, FEJ or equivalent policy; test consistency. Full mode refers to this declared fixed-lag approximation, not exact smoothing over all 60 frames.

### 3.5 Practical resource policy

- Implement an exact dense CPU reference for 1–4 points, 2–4 frames, 1–2 cameras, tiny images, with unique raw pixel latent keys.
- Implement sparse matrix-free GPU path using connected components of the state–latent-pixel factor graph and **exact per-component elimination** where feasible. Respect point ID and camera/frame/pixel keys, and retain cross-point couplings if different points reuse the same raw reference pixel.
- Use streaming/chunking only across **independent graph components**, never across a component edge. Assert graph-cut completeness before solving.
- Test the full real-scene point count (including scenes with >200k points) for peak GPU memory and runtime. If 32-GB GPU cannot satisfy the requested exact cross-component covariance, report `FULL_RESOURCE_LIMIT` with component sizes and memory; **never silently downgrade to simplified**.
- `--temporal-lag-frames L` has a minimum 2. References older than the active lag must be represented by an exact marginalized boundary factor or declared unusable with `REFERENCE_OUTSIDE_LAG` (do not silently treat old states as deterministic). Start with `L=3` as a computational test configuration, not a new physical tuning threshold.

## 4. Simplified mode: explicit approximation of the same model

Use exactly the same z-buffer, reference lifecycle, q warp, measured mask and interpolation support. Replace the full joint latent/reference system by:

```
Ref state x_tau: frozen at its previously accepted estimate
Ref clean patch ell_tau: use measured RGB-D image values
r = y_t - [I_tau(q); D_tau(q) + deltaZ]
H_t = [[-grad I_tau(q) Jpi_t, 0],
       [JZ_t - grad D_tau(q) Jpi_t, 0]]
R_eff = R_current + W(q) R_reference W(q)^T + R_warp
```

`W(q)` is the bilinear interpolation matrix operating on **raw reference pixel noise**; `R_reference` contains per-pixel sensor noise from the configured sensor calibration. This is more honest than treating the reference noise as zero, but simplified mode **neglects** the reference-state covariance, shared raw reference noise across different temporal factors and the correlation between current prior and reused reference measurements. Publish manifest field:

```
reference_uncertainty = "fixed_reference_state + marginal_sensor_noise_only;
                         ignores_temporal_and_cross_point_correlations"
```

Use the existing independent `6x6` fixed-prior IEKF information solve, but with new temporal rows. No `x_tau` correction. Preserve `P_plus=P_minus` for points without valid rows. `R_eff` must use current and reference noise consistently and **must not** be treated as a claim of exact covariance.

**Optional diagnostic only (not a required third mode):** `reference_noise=zero` deliberately sets `R_reference=0` to quantify the overconfidence caused by a perfectly known reference. It must be labelled `UNSAFE_ZERO_REFERENCE_NOISE_ABLATION` and disabled by default.

## 5. First-hit visibility, hard mask and reference lifecycle (shared)

### 5.1 Ownership cache

Implement `estimator/temporal_zbuffer.py`:

```
first_hit_pixel_owner(points, camera_rig, frame_geometry, dtype):
    # no RGBD measurements used to choose a rendered frontmost point
    # output owner_id[C,H,W], owner_depth[C,H,W], occupied[C,H,W],
    #        projected_uv[C,N,2], reason[C,N], counts, ties, conflict records
    # index from original point IDs, stable sort by (camera, pixel, z, ID)
```

Use CUDA/PyTorch segmented sort first; optimized fused CUDA later only if parity is preserved. `owner_id=-1` for uncovered pixels. Camera geometry may be computed for all points (occluders), even if only a subset will have active reference-based measurements. This is not the legacy per-point all-camera RGB-D assimilation.

### 5.2 Hard mask rules

- Current target pixel must satisfy `mask[c,t,p] == True`, valid metric depth and finite RGB.
- Every bilinear reference support pixel must satisfy `mask[c,tau,pa] == True`, valid metric depth and same-surface policy; no masked-out RGB-D interpolation, even with zero weights by accident.
- **Do not** assign RGB-D rows to rendered points over observed background. Evaluate hard foreground feasibility using the observed mask, without reading RGB/depth from that background.
- If a frontmost projected point lies over observed background, report `RENDERED_OUTSIDE_FOREGROUND`. If hard feasibility is enabled, its candidate update must not violate the foreground support constraint; do not simply remove the invalid row and claim lower objective.
- If observed foreground has no rendered owner, report `UNEXPLAINED_FOREGROUND` and preserve it for future backward discovery (outside this implementation).
- Hidden points do not get a mask or RGB-D correction from cameras in which they have no selected pixel. A mask-only repair must never move an occluded point due to an unrelated visible pixel.

### 5.3 Camera handover and reference initialization

```
NO_REFERENCE
   -> first currently owned, valid observed foreground pixel: BOOTSTRAP_PENDING
   -> after accepted frame solution: REFERENCE_READY (no same-frame temporal information)
REFERENCE_READY + visible and warped support valid
   -> TEMPORAL_MEASURED
REFERENCE_READY + occluded
   -> OCCLUDED_NO_ROW (retain reference with timestamp)
REFERENCE_READY + invalid/stale warp support
   -> REFERENCE_INVALID / NEED_BOOTSTRAP
```

A newly visible point-camera pair has **no** same-camera temporal factor on its first valid frame; merely using that frame to initialize a reference is not a Kalman measurement. The old camera may continue updating the global point. When both A and B have validated histories, both can update the same state. When camera A disappears, no old A observation is used as an active independent measurement; any A information already incorporated persists through the state prior.

When reusing an old reference after occlusion, confirm that reference support survives the projective warp and remains within the declared lag; otherwise rebootstrap. Do not use a hardcoded maximum gap or velocity gate. Reference validity is geometric, masked, statistical and resource-policy based.

After the **accepted** IEKF solution, update references using actual observed RGB-D and the accepted owner map. Do not refresh from rejected iterates. In full mode, new references are part of the same latent-image graph and share their observation variables when used subsequently; never copy their measured RGB-D into an independent noiseless constant.

### 5.4 Hard owner switches and consistency

Inside one linear solve, freeze point owner IDs and reference interpolation-cell identities. After an accepted nonlinear candidate, rebuild the z-buffer and reference support. At candidate acceptance, score the *same measured raw pixels* with explicit uncovered/mask feasibility; no merit by erasing high-residual factors. Record every owner switch and owner-depth difference; use stable deterministic tie-breaking. Since hard switches are nonsmooth, report their incidence separately and do not claim global differentiability.

## 6. Recommended new modules and public signatures

Create separately, keeping existing `legacy` untouched:

```
estimator/temporal_zbuffer.py
    first_hit_pixel_owner(...)
    compare_owner_maps(...)

estimator/temporal_reference.py
    ReferenceTable
    initialize_from_source_catalogue(...)
    propose_acquisitions(...)
    validate_reference_patch(...)
    commit_accepted_references(...)

estimator/temporal_rgbd_terms.py
    build_temporal_rows(...)
    temporal_prediction(...)
    temporal_jacobians_two_time(...)
    cpu_finite_difference_oracle(...)

estimator/temporal_full_graph.py
    UniquePixelRegistry
    TemporalWindowGraph
    add_raw_sensor_factors_once(...)
    add_warp_factors(...)
    add_process_factors(...)
    add_coupled_anchor_factors(...)
    linearize(...)
    solve_sparse_map(...)
    marginalize_expired_exactly(...)
    extract_current_marginals(...)

estimator/temporal_simplified_iekf.py
    update_points_temporal_simplified(...)

estimator/temporal_run.py
    run_temporal_free(...)

tools/evaluate_temporal_rgbd.py
    # post-hoc independent evaluation only

tools/compare_temporal_modes.py
    # compare baseline/full/simplified: numeric, timing, covariance, support
```

The implementation may consolidate tiny utilities but keep the `full` and `simplified` mathematical paths unmistakable. Do not rename old legacy classes to imply they implement the new model.

## 7. Integrate F1/B1 without breaking the existing pipeline

In `train.py`:

```
--observation-model {legacy,temporal-rgbd}           # default legacy
--temporal-mode {full,simplified}                   # used only with temporal-rgbd
--temporal-lag-frames L                            # 3 recommended for pilot, >=2
--temporal-reference-policy latest-valid           # required initially
--temporal-warp-noise {measured,config}            # measured requires saved empirical calibration
--temporal-warp-noise-config PATH                  # required for config policy; no hidden default
```

Reject invalid mode combinations; when temporal-rgbd selected, explicitly reject `--rendered-mask-measurement on` until a unified derived factor exists. Register every new module/config in `STAGE_SOURCES` and `source_hashes`. Change manifest fields from misleading `renderer_used=False`/`correspondences_used=False` to accurate per-mode entries, including `observation_model_version`, `noise_provenance`, `temporal_mode`, `lag`, owner semantics, marginalization approximation and original scene/calibration hashes.

In `free_projective_run.py`, replace the `update_points(...)` call **only** for explicit temporal-rgbd invocation. Reuse original canonical point ID order, endpoint arrays, `predict`, covariance precision and chronology. A running full window must not rebuild from scratch and forget the previous marginal prior. F1 forward uses 0→T; B1 uses T→0 and terminal anchors; temporal reference `tau` is the immediately previous valid frame **in processing order**, not necessarily a numerically smaller frame index. The q warp and signed `deltaZ` remain defined current minus reference.

**Downstream:** Existing coherence/rigid/joint stages may consume the temporal free outputs only after the exact same array names and shapes are verified. The legacy F2/B2 projective observation model is not changed by this contract. Mark downstream reports as `TEMPORAL_FREE_WITH_LEGACY_RIGID` until an explicit follow-on spec changes the rigid observation model. Never silently claim whole-pipeline temporal inference if only F1/B1 changed.

## 8. Numerical correctness & objective acceptance

### 8.1 IEKF optimization

- Fix the dynamic/process prior during a given frame's IEKF iterations.
- Build the first-hit owner map at the current nonlinear iterate. Freeze the owner and reference pixel 2x2 support during a sparse linear solve.
- Calculate Jacobians of the actual latent temporal prediction; no gradient of the *current* observed image substituted for the reference gradient.
- Apply the state and latent-pixel correction to a candidate; rerender/resort and recalculate candidate full objective and hard mask feasibility before acceptance.
- A candidate that only removes measurement rows is **not** automatically an improvement; maintain an explicit common-domain raw sensor likelihood plus warp factors and report `UNEXPLAINED_FOREGROUND` separately. Document what happens to factors when their owner switches (retire/reassign/switch-compatible likelihood). If this cannot be achieved exactly under hard assignment, mark the objective as a piecewise approximation and use conservative acceptance plus independent held-out diagnostics; do not claim a mathematically identical objective when factor sets differ.
- Damping/backtracking is acceptable for local nonlinear convergence (same prior, correct derivatives), but do not invent physical movement/speed gates. Report actual full objective, proposal objective, residual-gradient norm, accepted iterations, rejected iterations, switched pixels and convergence reason.
- Compute covariance once at accepted linearization, including reference uncertainty for `full`, and record local Gaussian approximation status.

### 8.2 Full covariance vs simplified covariance

- `full`: `P_current` is a **marginal** from the joint state+latent graph, including temporal/state/reference uncertainty and shared-pixel cross-correlation (under fixed-lag local linearization).
- `simplified`: `P_current` is independent per point, neglects temporal/cross-point correlations. Display diagnostics with an explicit `APPROXIMATE_COVARIANCE` label.
- `legacy`: original covariance behavior unchanged.

### 8.3 Statistical calibration

For each mode compute innovation mean/std, covariance/condition number, normalized innovation squared (NIS), posterior NEES **only where post-hoc GT is available**, and empirical coverage. A correlated batch residual vector needs the correct full `S` and degrees of freedom (`rank`), not “4 × number of naive rows” when correlations exist. Record Q and R provenance and any empirical Bayes calibration details.

## 9. Required tests (implement before actual datasets)

New tests should live in `tests/test_temporal_*.py`; run alongside `python -m unittest discover -s tests -t .`.

### Gate A — calibrated geometry and exact ownership

- Fixed 2-camera pinhole setup; known point positions/depths and original source IDs.
- Identity projection and source ownership at frame 0; deterministic point-ID tie resolution.
- Two points map to one pixel: only nearer point owns it, loser has zero direct Jacobian; swap depth, owner changes on rerender.
- Cross-camera candidate occlusion: a point may win in A and lose in B; camera-specific rows reflect that.
- Depth sorting parity CPU NumPy vs CUDA/PyTorch across random seeds and float64 (sort ties, pixel round convention).
- Tests for all points occluded: exactly predicted state and covariance, no artificial covariance contraction.

### Gate B — temporal warp & two-time Jacobian

- Exact constant-color translation (RGB Jacobian is zero, correct unobservability).
- Textured affine RGB and sloped depth patch translated 0.2 px, 1 px and several pixels with known synthetic positions; check residual, sign and units.
- Finite differences for `H_t` and `H_tau` for all 6 state columns (velocity columns zero at fixed frame pose); central differences within the same interpolation cell.
- Boundary and mask: 2x2 reference neighbor background, NaN, depth discontinuity, out-of-image -> explicitly invalid with finite dummy storage and zero information.
- Test frame `t` and backward direction `t<tau` with consistent depth and camera poses.

### Gate C — full latent graph correctness

- 1 point, 1 camera, 2 frames, 2x2 reference pixels: compare dense CPU full posterior mean/covariance with sparse GPU/torch solve.
- 1 point, 1 camera, 3 frames, **same raw reference pixel reused**: registry contains exactly one raw sensor observation factor; correct off-diagonal residual covariance after eliminating latent pixels.
- 2 points, 2 cameras, overlap in reference interpolation supports: same raw pixels are not independently counted; cross-point covariance matches dense oracle.
- Frame-0 canonical source depth is used **once**: compare coupled backprojection construction to independent generative oracle; deliberately duplicated anchor must fail test.
- Eliminate old frame via Schur complement: log determinant / marginal covariance parity versus solving the uneliminated graph; PSD/symmetry/finite checks and `lambda_min`.
- Compare dense oracle `Cov(x_t,x_tau)` and `Cov(x_t,ell_ref)` with extracted GPU trace blocks.
- Run 3-frame lag with sliding to 4 frames and check expected marginal prior; no reset/forget of correlated reference information.
- NIS/NEES Monte Carlo with correlated reference noise; full should pass declared confidence intervals under its own *correctly specified* synthetic generative model (not an arbitrary hard-coded outcome).

### Gate D — simplified is a simplification, not a new observation

- Identical winner map, warp coordinates, patch validity and Jacobian as full on same states.
- With `P_tau=0`, no shared reference noise between factors, and negligible latent uncertainty, full and simplified state correction agree to float64 tolerance.
- With reference noise and reused pixels, simplified may be overconfident: assert the expected covariance distinction and label it, not exact equality.
- Legacy mode numerical parity for one same-seed run; existing tests all pass.

### Gate E — camera handover

- 2 fixed cameras, moving point A-only -> A+B -> B-only; point global ID/state remains unchanged.
- First B-visible frame is reference bootstrap only, **not** zero-noise/zero-residual update.
- Next B frame yields a same-camera temporal residual; no comparison against A RGB.
- Existing A state information survives via prior; independently observed B can update after bootstrap.
- Sudden occlusion and reappearance: reference invalidation/rebootstrap or valid reuse, no background image access.

### Gate F — nonlinear IEKF / adversarial cases

- Wrong owner, changed owner, large pixel shift, static textured scene: report objective, correctly reject/accept, no silent disappearance of factors.
- Store + replay a traced update with all latent pixel keys, weights, prior blocks, `H_t/H_tau`, `g`, Lambda, step, posterior/cross-covariance.
- Perturb camera image sample by `1e-12` px and report solution sensitivity; exact agreement of recomputed owner map and local Jacobian where support is unchanged.
- Compare 1 vs 2 vs several cameras using same trajectory; no automatic cross-camera color bias (temporal refs stay camera-specific).
- Time and CUDA memory bounds; deterministic results under point chunk sizes **only when chunks do not split a graph component**.

If a Gate fails, fix it **before** running large synthetic scenes. Output `tests/temporal_test_results.json` with each check, tolerances, runtime, CPU/CUDA pass and Git commit. A blank `PASS` without numerical witnesses is unacceptable.

## 10. Required diagnostics and artifacts

Preserve `free_{forward,backward}/numeric/frame_XXXX.json`, `timeline.csv`, `trace_frame_XXXX.json`, `posterior_xyz.npy`, `velocity.npy`, canonical point metadata and posterior variance arrays.

Add **small, targeted** temporal diagnostics (no per-frame dense snapshots by default):

```
free_<dir>/temporal/
  config.json
  calibration.json                        # sensor and warp noise provenance
  ownership_summary.csv                  # per camera/frame selected/occluded/switches
  reference_lifecycle.csv                # bootstrap, valid, occluded, reused, invalid
  innovation_summary.csv                 # RGB, depth, NIS, rows, gradient norms
  covariance_summary.csv                 # SPD, cross-time norms, marginalization
  timing_memory.csv                      # z-sort, sampling, assembly, solve, Schur
  traced_points/<scene>_<point_id>.json   # only IDs explicitly requested
  debug_frames/frame_0001_cam0.png       # only requested debug frames
  report.md
```

Each traced point/frame: original source `(camera,pixel)`, canonical index, each current winning `(camera,pixel)`, both reference/current 2x2 raw image pixels and masks, interpolation weights, `q`, gradients, projected depth, winner and second depth, first-hit conflict point ID, `s^-/P^-`, previous state and reference state, `H_t/H_tau/H_ell`, residual, per-camera information, reference reuse count, crosscovariance and Schur blocks (full), conditional covariance (simplified), candidate/accepted objective, convergence, velocity increment, and reason for each excluded row.

For `Storage 8444/45588`, additionally save the exact camera-0 collision at pixel `(282,237)` and the next 10-frame ownership history. For USB 67827 inspect cameras 5/6/4 and late cam-4 reference bootstrap. For USB moving 87290 and 85696 record frames 1–10 and whether actual image rows support the state; do not call prediction-only trajectories successful.

At `--debug-frames`, generate compact overlay panels per camera: observed RGB, observed metric depth, hard mask, first-hit owner index colors, predicted prior/post positions, referenced warped RGB/depth, RGB/depth residual, uncovered foreground and reference-bootstrap locations. No need to save every source frame duplicate.

## 11. Synthetic scenes and hypotheses (GT evaluation separated)

Prioritize the real 8-camera SAPIEN ray-cast directories; inspect availability, do not assume exact path exists. Use these real point IDs only when `pcd-subsample-factor=1` and canonical index mapping matches the archived fifth-pass run.

| Scene | Why critical | Named points / frames | Required report |
|---|---|---|---|
| `storage_45135` | Static drift & pixel collision | 8444, 45588, 49199, 262894, 205921; frames 0–10 | Source ownership, cross-camera temporal offsets, false velocity, both competing trajectories |
| `USB_100109` | Static crease, camera handover, moving-point loss | 67827, 87290, 85696; frames 0–10 | Cam5/6/4 bootstrap history; moving observation coverage; moving GT post-hoc error |
| `blade_103706` | Persistent failure unexplained by fifth pass; motion baseline | 14104, 27016; frames 0–10 and endpoints | Observation support, numerical rank, drift vs actual physical motion |
| `stapler_103111` | Additional static-runaway stress test | first 11 frames plus final frame when feasible | Static displacement distribution; accepted/rejected motion and observability |

A practical order: tiny synthetic oracle -> 3 frames on Storage -> 11 frames Storage & USB -> 11 frames Blade -> 11 frames Stapler -> **only after gates** all 60 frames on Storage, USB and Blade. Full mode may be expensive; do not silently substitute simplified. Report `FULL_RESOURCE_LIMIT` separately from a numerical or scientific failure.

For every scene/mode compute: mean/median/p95/p99/end static displacement using GT **only in the separate evaluation process**, fraction static >10cm, tracked moving-point 3D error, number of observations per point/camera, no-row fraction, frame-1 velocity false boost, owner collision count, camera handover latency, innovation calibration, SPD/covariance consistency, wall time and peak CUDA memory. Avoid interpreting zero static displacement with zero rows as a success.

## 12. Clear success/failure conditions

**Scientific validity requires all:**

1. Legacy mode unchanged and existing test suite passes.
2. Full dense-versus-sparse oracle reproduces cross-time and shared-pixel covariance, including coupled source initialization.
3. Frame-0 source provenance never silently erased; point IDs stable; camera handover without cross-camera RGB equality assumption.
4. Correct no-measurement behavior for entirely occluded points, with no background RGB-D inputs.
5. Full and simplified share identical warp, visibility and measurement semantics; only uncertainty differs.
6. Full method's published covariance is a real marginal of the retained joint graph, not inverse conditional Hessian.
7. Objective acceptance does not claim improvement solely by eliminating difficult image rows.
8. The critical failures are measured quantitatively, including static **and** moving, with honest uncertainty and visibility attribution.

**Do not equate test completion with success on real scenes.** If a failure cannot be resolved without fundamentally changing the first-hit association or local translation model, report it plainly with numerical counterexamples, rather than inventing more thresholds.

## 13. Deliverables

- Working code for `--observation-model temporal-rgbd --temporal-mode full|simplified`, with baseline preserved.
- Dense CPU oracle + GPU implementation + complete tests.
- Self-contained `docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md` with actual implemented equations, lag treatment, factor graph, Schur derivation, source-depth handling, covariance approximations, corner cases and provenance.
- `reports/temporal_rgbd_implementation.md`, including code locations, measured test numbers, known remaining issues, CUDA profile, and commands used.
- `reports/temporal_rgbd_critical_scenes.md`, including per-scene per-mode tables, objective and uncertainty checks, GT-free diagnostic versus GT-posthoc columns.
- `tools/run_temporal_rgbd_critical.sh` supporting `--data-root`, `--output-root`, `--mode`, `--frame-limit`, `--direction`, `--scenes`, `--point-ids`. Must print exact commands, preserve outputs, fail on missing scene, support spaces in paths, avoid overwriting existing runs and never consume GT during inference.
- Update README, source hashing, manifest/resume validation and code map.
- **Commands for the human operator**, with no long-run launches by Codex unless explicitly requested.

## 14. Commands Codex should use while developing (small only)

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
python -m unittest discover -s tests -t . -v
python -m unittest tests.test_temporal_warp tests.test_temporal_full_graph -v
python train.py --help | grep -E 'observation-model|temporal-mode|temporal-lag-frames|temporal-warp'
```

Make tests discoverable as named modules, or adapt the module names to actual created files. Start with independent in-memory CPU toy scenes, no external datasets.

## 15. Real-scene preflight before human execution

Data root used in prior validated reports (the human operator must verify it exists):

```
/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear
```

Expected scene directories: `storage_45135`, `USB_100109`, `blade_103706`, `stapler_103111`. These paths are from the repository's prior reports, not checked on the local operator workstation in this chat. Check `rgb/` and `metadata/` presence before invoking `train.py`; the loader accepts a scene folder or root of scenes.

## 16. Minimal operator command — run critical 11-frame sequences (AFTER Codex implementation)

The `--observation-model`, `--temporal-mode`, `--temporal-lag-frames`, and `--temporal-warp-noise` arguments **do not exist in the legacy CLI**; Codex must implement and document them before these commands work. In this contract, Codex should implement `--temporal-warp-noise measured` and write the estimated/modelled variance and its provenance to the manifest. If a calibrated warp noise cannot be estimated safely, fail with a clear message requiring `--temporal-warp-noise-config PATH`; do not invent a variance. The syntax below is the required new CLI contract.

```bash
set -euo pipefail
REPO="$HOME/Joint3DGS/RGBDekf"
DATA_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
OUT_BASE="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf"
RUN_TAG="temporal_rgbd_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$OUT_BASE/$RUN_TAG"
PY="$HOME/miniconda3/envs/rigeuler3dgs/bin/python"

cd "$REPO"
for s in storage_45135 USB_100109 blade_103706 stapler_103111; do
  test -d "$DATA_ROOT/$s/rgb" && test -d "$DATA_ROOT/$s/metadata" || {
    printf 'Missing scene: %s\n' "$DATA_ROOT/$s" >&2; exit 1;
  }
done
mkdir -p "$OUT_ROOT"

for mode in full simplified; do
  for scene in storage_45135 USB_100109 blade_103706 stapler_103111; do
    case "$scene" in
      storage_45135) ids='8444,45588,49199,262894,205921' ;;
      USB_100109) ids='67827,87290,85696' ;;
      blade_103706) ids='14104,27016' ;;
      stapler_103111) ids='' ;;
    esac
    out="$OUT_ROOT/${scene}_${mode}_11frames"
    extra=()
    if [[ -n "$ids" ]]; then extra=(--debug-point-ids "$ids"); fi
    "$PY" -B train.py \
      --scene "$DATA_ROOT/$scene" \
      --output "$out" \
      --device cuda \
      --direction forward \
      --stop-after free \
      --frame-limit 11 \
      --pcd-subsample-factor 1 \
      --covariance-precision float64 \
      --sensor-noise measured \
      --observation-model temporal-rgbd \
      --temporal-mode "$mode" \
      --temporal-lag-frames 3 \
      --temporal-warp-noise measured \
      --rendered-mask-measurement off \
      --debug light \
      --debug-frames 0,1,2,3,5,7,10 \
      "${extra[@]}" \
      --no-evaluate --no-qa \
      2>&1 | tee "${out}.log"
  done
done

printf '\nResults: %s\n' "$OUT_ROOT"
```

**Important:** `--no-evaluate` keeps GT entirely outside inference. Once inference completes, call the independent `tools/evaluate_projective_gt.py` or new `tools/evaluate_temporal_rgbd.py` on those outputs in a separate invocation; do not embed GT into the runner. If local point ID provenance differs from archived traces, report corrected mappings instead of silently assuming labels. This is a *human-executed* command, not an instruction to Codex to launch unattended full runs.

## 17. Full 60-frame command (run only after all validation gates)

```bash
set -euo pipefail
cd "$HOME/Joint3DGS/RGBDekf"
DATA_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
OUT_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_full_60_$(date +%Y%m%d_%H%M%S)"
PY="$HOME/miniconda3/envs/rigeuler3dgs/bin/python"
mkdir -p "$OUT_ROOT"
for scene in storage_45135 USB_100109 blade_103706; do
  test -d "$DATA_ROOT/$scene/rgb" && test -d "$DATA_ROOT/$scene/metadata"
  "$PY" -B train.py \
    --scene "$DATA_ROOT/$scene" \
    --output "$OUT_ROOT/$scene" \
    --device cuda --direction both --stop-after free \
    --pcd-subsample-factor 1 --covariance-precision float64 \
    --sensor-noise measured \
    --observation-model temporal-rgbd --temporal-mode full \
    --temporal-lag-frames 3 --temporal-warp-noise measured \
    --rendered-mask-measurement off \
    --debug light --debug-frames 0,1,2,3,5,10,30,59 \
    --no-evaluate --no-qa \
    2>&1 | tee "$OUT_ROOT/${scene}.log"
done
```

Do not use `--frame-limit 11` for a B1/backward endpoint experiment intended to start at original frame 59: this flag truncates the dataset, so its backwards anchor would be frame 10. Use all 60 frames for real endpoint-59 backwards tests, or provide a separately documented last-frame slicing mechanism.

## 18. Operator post-run checks

```bash
cd "$HOME/Joint3DGS/RGBDekf"
python -m unittest discover -s tests -t . -v
# For each output run, inspect the manifest, full/simplified mode,
# full covariance/marginalization status, acceptance reason counts and timings.
# Then use the independently implemented comparison tool:
# python tools/compare_temporal_modes.py --root '/absolute/output/root'
# python tools/evaluate_temporal_rgbd.py --run '/absolute/run' --scene '/absolute/scene'
```

The comparison/evaluation tool flags are **to be implemented by Codex**, not claimed existing. It must not require GT for numeric temporal diagnostics. Keep the archive point-level case reports separate from aggregate scene metrics.

## 19. Required final report from Codex

Provide in one Markdown report:

1. Exact edited file list, source line ranges, Git SHA / dirty status and how baseline parity was verified.
2. The full generative observation model and exactly how source-depth double counting was prevented.
3. Dense CPU vs GPU mean and covariance discrepancies; Schur/marginal identity; JVP/VJP and finite-difference table.
4. Demonstration that one raw source pixel is used **once** in the full sensor likelihood even if used as a reference in multiple factors.
5. `full` versus `simplified` covariance and error comparison on toy and archived critical points, including observed support. Provide evidence rather than describing intended behavior.
6. Source/camera handover event traces, first-hit conflict and changes of ownership, annotated stored original point IDs.
7. GPU runtime, memory and graph component sizes; clear failure rather than silent fallback if full cannot scale.
8. Exact human commands for both the 11-frame pilot and subsequent full 60-frame evaluation; clearly state which runs actually completed, if any.
9. Open limitations: depth sort != physical correspondence, local translation warp fails under strong rotation/deformation, mask silhouettes, unlabeled static ambiguity and finite-lag approximation.

**Stop implementation only after these artifacts are produced and the relevant tests actually execute and pass.** For any genuine blocker, supply the error, failing numerical witness and smallest reproducible case. Do not claim a run or test result that did not occur.
