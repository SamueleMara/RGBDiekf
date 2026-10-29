# CODEX IMPLEMENTATION CONTRACT — Rendered Binary Pseudomask as an IEKF Image Measurement

**Repository:** `SamueleMara/RGBDiekf` (local checkout: `~/Joint3DGS/RGBDekf`)  
**Authoritative target:** the CURRENT LOCAL WORKTREE, including Codex's UNCOMMITTED visual-hull changes.  
**Verified public reference:** `main`, commit `9a9f16f44165bb757739a5253321d910f62e2c1f` (`Drift Debug Update`, 2026-10-28); **the local worktree is ahead of this commit**.  
**Target stage:** F1/B1 independent prior dynamics for point-only 6D RGB-D IEKF; F2/B2, coherence, joint and support consume the resulting trajectories unchanged.  
**This contract REPLACES** the previous `RGBDiekf_Multiview_Visual_Hull_Constrained_IEKF_Codex.md` **hard-geometry approach** and the subsequent `RGBDiekf_sparse_hard_geometry_performance_fix_for_Codex.md` proposal. Do **not** optimize or retain the expensive hard-QP/EP inference path.

## 0. The user's exact measurement definition (do not reinterpret it)

At frame `k=t+1`, for every calibrated camera `c`:

- The observed mask image is the **actual** binary foreground mask `M_obs[c,k,u,v]` already loaded from the dataset (`0=black/background`, `1=white/foreground`).
- Predict/transport **EVERY** tracked 3D point using its own `[x,v]` state, whether directly RGB-D visible or occluded.
- Render/project **EVERY** transported point as WHITE against BLACK, producing a new **binary pseudomask image** `M_hat[c,k]`; the point's RGB or opacity is not used. Do **NOT** exclude a point because of a z-buffer, `SELF_OCCLUDED`, `DUPLICATE_PIXEL_SUPPRESSED`, or rejection of its direct RGB-D row.
- The image innovation is **literally** `r_M = y_obs - y_hat`, where `y_obs=vec(M_obs)` and `y_hat=vec(rendered_white_points)`. This is an **additional image measurement channel IN the existing IEKF** alongside the actual RGB-D observations.
- **One-sided measurement row selection:** we only want to penalize *rendered white where the measured mask is black*. Keep image measurement rows only for `M_obs[c,k,u,v] == 0`. Missing predicted white within an observed white pixel is NOT an error. Thus at selected rows `r_M=0 - y_hat`.
- **Do not substitute** a signed-distance-to-boundary measurement `d(pi(x))`, a pointwise visual-hull inequality, a pseudo-depth free-space inequality, a censored-probit observation, a KKT projection of the estimate, an auxiliary silhouette loss, or a post-hoc optimization for this specified image innovation.
- The measured image is used ONCE per frame in the fixed-prior IEKF; iterative re-rendering and relinearization must not count it as independent new evidence every iteration.
- No rigid motion coupling assumption, predefined number of parts, static/dynamic labels, velocity clipping, maximum displacement, arbitrary physical speed threshold, or transfer of velocity from a visible neighbor to a hidden point.
- **Ground truth NEVER enters inference.** Use GT in separate evaluation only.

This task is specifically to replace a **30–64 seconds/frame** (some frames worse) QP/EP hard-geometry experiment with a sparse, direct **rendered-image IEKF measurement update**. The previous baseline was approximately **1.7–1.8 seconds/frame on storage**. These numbers come from the uploaded Codex log and are comparison context, not a promised new benchmark.

## 1. First: inspect the current worktree and preserve non-geometry fixes

Before touching files, run locally and record the output in the final report:

```bash
cd ~/Joint3DGS/RGBDekf
git status --short
git diff --stat
rg -n 'geometry.constraints|geometry_constraints|geometry_posterior|geometry_depth|ep_position_moments|reduced_qp|visual_hull_constraints|constrained_point_update|truncated_position_moments|geometry_trace|--geometry' .
```

Make an independent **git commit or diff/patch snapshot** of the working geometry code before removing the legacy path (respect existing uncommitted user changes; never hard-reset or discard unrelated work). Retain and do NOT revert the independently discovered `scene/dataloader.py` half-pixel synthetic camera-intrinsics correction, original RGB-D consistency/visibility fixes, and GT evaluator fixes. Do not modify any ground-truth evaluator as part of this feature unless necessary for a failing, independently reproducible test.

The public commit has no `visual_hull_constraints.py` / `constrained_point_update.py` / `truncated_position_moments.py`; the Codex report says those were created locally and integrated into F1/B1. The **real local source is authoritative**. Identify all references before deleting anything.

## 2. Existing code — required surgical insertion points

Symbol/line mappings below are accurate for public reference `9a9f16f`; **line numbers may differ in Codex's local worktree**.

| File / symbol | Existing purpose | Required replacement/addition |
|---|---|---|
| `estimator/projective_sensor.py`, `MeasuredFrame` (roughly lines 54–74) | Holds real `frame.mask[C,H,W]`, `rgb`, metric depth, validity | Keep the original mask as the image observation `y_obs`. Reuse dimensions and camera/frame identity; do **not** generate or use a boundary-distance measurement in the new mode. |
| `estimator/projective_camera.py`, `CameraRig.project` / `project_rows` (roughly 62–90) | World-to-camera calibrated `uv,Z,Ju` | Reuse projection and exact `Ju`, including corrected synthetic intrinsics and camera coverage semantics. |
| `estimator/projective_visibility.py`, `visibility`, `apply_frozen_assignment` | Frontmost/duplicate selection for direct RGB-D pixels | **Unchanged for RGB-D.** Mask renderer must ignore this exclusion and project ALL points. |
| `estimator/projective_terms.py`, `build_terms` | Existing direct RGB-D `r, H, R`, row admission | Preserve RGB-D equations, invalid row codes, existing noise/gate options. No mask-as-depth substitution. |
| `estimator/point_kalman_cuda.py`, `predict` (roughly 75–78) | `F(dt)`, signed-time `Q(dt)`, prior propagation | **Unchanged.** |
| `estimator/point_kalman_cuda.py`, `PointMeasurement` (roughly 164–225) | RGB-D `Ixx[N,3,3]`, `gx[N,3]`, `rTWr`, `nrows`, camera reasons | Keep unchanged where possible. Add a distinct image-mask measurement builder and its report fields, not fraudulent RGB-D row counts. |
| `estimator/point_kalman_cuda.py`, `update_points` (roughly 228–325) | Existing 6D per-point IEKF; local hard geometry was added here | Introduce a separate clearly named `rendered_mask_iekf` branch assembling **the SAME fixed-prior IEKF normal equations** with the sparse mask-image Jacobian. Preserve original RGB-D-only path for `off`. Remove all hard-QP/EP calls from production path. |
| `estimator/free_projective_run.py`, `run_free` (roughly 48–130) | F1/B1 frame loop and `update_points` | Re-render pseudomasks from transported points at each IEKF linearization; pass observed real mask. Save measurements/debug reports. Support forward and backward using real signed `dt`. |
| `estimator/free_projective_run.py`, `_store`, `_frame_numeric` | Output arrays, statuses, per-frame JSON | Add separate mask-image metrics/counters; preserve semantics of legacy `observed` = DIRECT RGB-D observed, `nrows`, `nis`, and `camera_reason`. Mark mask-only corrected points separately. |
| `configs/projective_rgbd.json` | Current RGB-D policies | Replace local hard-geometry settings with documented `rendered_mask_measurement` config. Do not silently change RGB-D policies, covariance, or priors. |
| `train.py`, `STAGE_SOURCES`, CLI, stage call, `source_hashes`, resume checks | Orchestrates all five stages; currently hashes the free-stage sources | Register new module(s) in source hashes; pass explicit mode, rendering/noise configuration; ensure resume rejects incompatible measurement models. |
| `debug/projective_debug.py` / debug assets (inspect exact local names) | Numerical and visual trace outputs | Show the **actual measured mask**, **rendered binary white-point pseudomask**, **image-space false-positive pixels**, plus numerical measurement/Jacobian diagnostics. |
| `tests/` | Existing F1/B1 and downstream test suites | Add strict rendering, Jacobian, sparse joint solver, uncertainty and full-pipeline tests. |

Recommended new modules:

1. `estimator/rendered_mask_measurement.py`: differentiable WHITE-POINT image renderer, exact binary rasterizer for audit, one-sided image innovation, sparse pixel ↔ point contributor map, analytic renderer Jacobian, `Jv` and `Jᵀw` operators.
2. `estimator/rendered_mask_iekf.py`: joint sparse image-information IEKF solver, fixed-prior nonlinear iteration, analytic velocity recovery, covariance approximation contract and diagnostics.
3. `tests/test_rendered_mask_measurement.py`, `tests/test_rendered_mask_iekf.py`.

Prefer using existing GPU primitives and measured-frame arrays. Avoid unnecessary new packages. Any Triton/CUDA implementation must have an independently tested PyTorch CPU/float64 reference.

## 3. Exact image measurement definition and its derivative

### 3.1 The authoritative binary forward measurement

Let point `i` have predicted or iterated world position `x_i`, camera projection `u_ci = pi_c(x_i)`, and camera image dimensions `W,H`. For a **point-center white renderer**, define a deterministic pixel ownership rule (nearest-pixel-center with a documented tie break, consistent with the rig's pixel convention):

```text
B_c(x)[v,u] = OR_over_all_points_i 1{ raster_pixel(pi_c(x_i)) == (u,v) }
```

- Binary `B` MUST be an actual `H x W` 0/1 image, independently produced per camera.
- No visible-surface z-buffer for this mask channel.
- If projected point is behind the camera or outside the image, it has NO mask observation from that camera (do not fabricate black padding). Continue checking other cameras.
- Track the number of projected points (including hidden), the number of in-frame points, and the exact count of `B=1` in observed background.
- For a finite **specified** projected footprint, the OR is over its actual covered pixels, not just its center. Do not invent Gaussian 3D surfels, normals or meshes. The initial implementation should be point-center with a one-pixel nominal footprint.

Observed image, predicted image and raw image innovation:

```text
y_obs[c,k] = vec( M_obs[c,k] )           # actual binary image
y_hat[c,k] = vec( B_c(x) )               # actual binary pseudomask
r_raw[c,k] = y_obs[c,k] - y_hat[c,k]      # EXACT requested difference
```

Select image measurement rows for `M_obs==0` (and image-domain-valid):

```text
r[c,k,u] = (y_obs[c,k,u] - y_hat[c,k,u])  for M_obs[c,k,u] == 0
```

This is **one-sided**, because missing white points inside actual foreground are not penalized. It is not a full IoU/Dice/object-area-matching loss.

### 3.2 A hard 0/1 rasterizer has no useful IEKF Jacobian

For a binary nearest-pixel rasterizer `∂B/∂x` is zero almost everywhere and discontinuous at pixel crossings. A naive `H=0` cannot correct a point even with residual `r=-1`. Do not implement the literal hard 0/1 image as if autodiff will provide a correct derivative.

**Required representation for the IEKF linearization:** an **anti-aliased, differentiable rendered image of white points** using a physically explicit pixel coverage rule (not a distance-to-observed-mask residual). Initially use a bilinear point splat onto the four neighboring pixel centers (the natural pixel-grid interpolation, with exact derivative almost everywhere). For point `i`, camera `c`, pixel `p`:

```text
a[c,p,i](x_i) = max(0,1-|u_i-u_p|) * max(0,1-|v_i-v_p|),   0 <= a <= 1
```

Exactly 2x2 nearby pixel centers receive coverage at generic subpixel coordinates. Define differentiable white-point union occupancy:

```text
m_hat[c,p](X) = 1 - PRODUCT_{i contributes to pixel p}(1-a[c,p,i])
```

- The RGB-D scene remains **point-only**; no photometric color, depth visibility, 3D Gaussian renderer, mesh or per-point normals.
- The differentiable image `m_hat in [0,1]` is an **anti-aliased rendering of the same white points**. Its target `y_obs` remains the **real original binary image**. This is a standard local image-measurement surrogate, not a substitute distance observation.
- Preserve a **separate hard binary pseudomask** `B` for exact user-requested visual verification. Log both hard-image outside-foreground count and anti-aliased IEKF image innovation; do not falsely claim those are numerically identical away from raster boundaries.
- For a pixel with contributors `I_p`, the analytical derivative is

  `∂m_hat[c,p]/∂x_i = PRODUCT_{j in I_p, j≠i}(1-a[c,p,j]) * (∂a[c,p,i]/∂u_i) * J_pi[c,i]`,

  where `∂a/∂u_i` is the 1×2 pixel-coverage gradient and `J_pi` is 2×3. The derivative is zero when `i` is not a contributor. A zero/subgradient convention at the finite number of non-differentiable grid-aligned locations must be documented/tested.
- Compute the product-excluding-self **without dividing by `1-a` when `a=1`** (that would create `0/0`). Use a stable segment prefix/suffix or another proven equivalent. Deterministic tie and reduction policies required.

If an alternative soft renderer is proposed, it must still literally render **white points into the same image pixels**, keep the raw `y_obs-y_hat` innovation and pass finite-difference Jacobian tests. No hidden SDF/probit replacement.

### 3.3 The one-sided image observation for IEKF

The actual binary measured image is `M_obs`; select rows by `M_obs=0`. At these pixels:

```text
r_mask[c,p] = 0 - m_hat[c,p](X_j)
J_mask[c,p,i] = ∂m_hat[c,p](X_j) / ∂x_i
```

Equivalently `r_mask = (y_obs - m_hat)[y_obs == 0]`. Ignore `y_obs==1` rows entirely in mask information accumulation. The original full-image raw residual can still be displayed for debugging, but **must not be assimilated on foreground pixels**.

**Do not filter out a pixel because RGB-D rejected it**. These are different observation channels. Only use measured-mask image coverage, camera validity, pixel-domain availability and the explicit one-sided selection.

### 3.4 A genuine limitation to test and REPORT (do not conceal it)

A pure one-sided binary image difference does **not** always provide a direction back toward the observed foreground: if a white point and its entire differentiable footprint lie far away in a uniform black region, the image residual is nonzero but its image-gradient step can vanish, have only periodic subpixel derivatives, or lead to a locally stationary erroneous rendering. This is a mathematical limitation of **this exact measurement**, not a failure of GPU optimization.

Add an explicit regression fixture of a point far outside the true silhouette; measure `||JᵀR^-1 r||` and whether the IEKF correction is informative. **Do not secretly introduce boundary-distance forces, velocity caps, huge arbitrary splats, or a secondary attraction loss** to make that test pass. Report those cases as image-measurement locally unobservable and retain the prior/RGB-D behavior; later changes would require a separate user decision.

## 4. Direct IEKF equations — same RGB-D prior, new mask image rows

### 4.1 Dynamics remain independent

For each point independently:

```text
s_i = [x_i (world meters); v_i (world meters/second)]  in R^6
s_i^- = F(dt) s_i^+
P_i^- = F(dt) P_i^+ F(dt)^T + Q(dt)
```

Retain negative `dt` for B1, the physical timestamps, and all existing process-noise semantics.

At a **fixed frame** the measurement is coupled in image space if two points render into the same pixel. The *prior dynamics* stay point-independent; **the exact joint image posterior does not remain point-independent**. Never call a block-diagonal approximation an exact shared-image IEKF.

### 4.2 Stack actual image observations

Form a single fixed-prior IEKF measurement model:

```text
y_obs = [ y_rgbd_obs ; vec(M_obs[C,H,W]) on measured-background rows ]
h(X)  = [ h_rgbd(X)   ; vec(m_hat_white_points(X)) same rows ]
r(X)  = y_obs - h(X)
H(X)  = ∂h(X)/∂S
```

The direct RGB-D Jacobian is only for actual foreground surface points admitted by the **existing** `projective_visibility.py`/`projective_terms.py` pipeline. The new rendered-mask image Jacobian includes **every transported point in coverage**, regardless of whether its RGB-D row was valid.

A jointly Gaussian surrogate observation covariance `R = blockdiag(R_rgbd, R_mask)` makes the usual IEKF information formulation available. The mask is originally binary; this Gaussian pixel-coverage model is an **explicit approximation**, not a claim of exact Bernoulli noise. `R_mask` must be positive and declared/calibrated in mask-intensity² units. Support a *per-camera/frame pixel variance* when the source permits it. Do not silently choose or fit an arbitrary strong weight from GT results. If segmentation and RGB share error sources, document double-counting/correlation and report the assumption; don't falsely claim independent measurements.

### 4.3 Exploit exact 6D→3D elimination (no full 6N dense solves)

Measurements depend on current positions, not current velocities. For the pointwise predicted Gaussian partition:

```text
P_i^- = [ S_i    B_i^T ]
        [ B_i    V_i   ]
S_i = P_xx,i^- (3×3), B_i=P_vx,i^- (3×3)
L_i = B_i S_i^-1
C_i = V_i - B_i S_i^-1 B_i^T
```

For any selected updated position `x_i`, the prior-conditional velocity is exactly

```text
v_i(x_i) = v_i^- + L_i (x_i - x_i^-).
```

This permits a **3N-position** IEKF solve while keeping all 6D point states. Let `δX ∈ R^(3N)` be the correction from current IEKF iterate `X_j`. Let `B0=blockdiag(S_i^-1 + I_rgbd,i)` and `J` be the image-renderer Jacobian on observed-background image pixels. Let `r_M=y_obs - m_hat(X_j)` and `r_rgbd` be existing admitted RGB-D innovations. The Gauss–Newton fixed-prior IEKF normal equations are:

```text
A δX = b
A = blockdiag(S_i^-1 + I_rgbd,i) + J^T R_mask^-1 J
b = -blockdiag(S_i^-1)(X_j - X^-) + g_rgbd + J^T R_mask^-1 r_M
```

where `I_rgbd,i` and `g_rgbd,i` come from existing projective RGB-D code and are evaluated at `X_j`. Solve the **joint** system for `δX`, then:

```text
X_{j+1} = X_j + δX
v_{i,j+1} = v_i^- + L_i (x_{i,j+1} - x_i^-).
```

Signs MUST match the existing `r=y_obs-h(x)` and the existing RGB-D `gx` accumulation. Do not invert an image-sized measurement covariance. Do not perform hard-geometry QPs or EP after this solve.

At convergence, `A` is a Gauss–Newton information approximation to the combined observation model. Nonlinear renderer/assignment changes require iterative rerendering with **the same original prior** and **the same actual measured mask**. If using globalization, evaluate actual current combined Gaussian-surrogate likelihood/prior; do not introduce an external silhouette objective.

### 4.4 Exact sparse multiplication, not dense pixel→point matrices

Each projected point has at most four nonzero anti-aliased pixel contributions per camera in the initial 2x2 splat. With `C=8`, this is at most 32 **contributor entries per point** (subject to FOV), before handling pixel overlap.

Define sparse image Jacobian actions, without ever constructing `H_mask` dense:

```text
# J: from per-point 3D increments to measured-background pixels
(J v)[p] = SUM_{i affecting p} J_mask[p,i] @ v_i

# J^T: from measured-background pixel scalars to point increments
(J^T w)[i] = SUM_{p affected by i} J_mask[p,i]^T * w[p]

# Information-matrix operator (matrix-free):
A(v) = blockdiag(S^-1 + I_rgbd) v + J^T( R_mask^-1 (J v) )
```

Use a batched GPU matrix-free **preconditioned conjugate gradient (PCG)** solver with a 3x3 per-point block-Jacobi preconditioner. The diagonal preconditioner is:

```text
A_diag,i = S_i^-1 + I_rgbd,i + SUM_p J_mask[p,i]^T R_mask[p]^-1 J_mask[p,i]
```

Factor these 3x3 SPD blocks with the existing precision/checking utilities. PCG updates all points jointly and retains overlap-induced cross-point terms; DO NOT silently remove `J_i^T R^-1 J_j` for `i≠j`. Never form a dense `3N×3N`, `HW×HW`, or `N×N` matrix.

Implementation details:

- Aggregate per-camera/per-pixel contributions with GPU scatter/segmented operations; `Jv`/`Jᵀw` must preserve all overlaps, including many points in the same pixel.
- Do not split independent point chunks in a way that drops cross-chunk contributors to the same pixel. Camera/buffer streaming is allowed if the **global pixel reduction remains correct**.
- Reserve image residual/Jacobian rows only at measured-background pixels **touched by a projected white-point footprint**; all other selected pixels have identically zero prediction and derivative and can be omitted **exactly**. This is the main image sparsity optimization.
- If no mask contributor occupies a measured-background pixel, the mask term is zero by construction.
- Use float64 numerical reference; production mixed/float32 only after unit tests. Include deterministic-or-specified reproducibility of parallel reductions.
- For a mask measurement with **zero informative rows**, return the original RGB-D IEKF solution and covariance path. The `off` path must stay bit-identical when possible.
- PCG stop via linear-system relative residual/energy criterion justified by floating-point accuracy and the requested IEKF tolerance, not a physical motion/speed threshold. Monitor actual residual and failure; do not silently accept stagnation.
- Reuse the prior's 3x3 Cholesky factor, the 3x3 RGB-D blocks and the sparse pattern within a linearized iteration. Rebuild only when point positions/pixel contributors change.
- Avoid one Python loop per pixel or point, and avoid hundreds of per-point QP/EP kernel launches. Profile and optimize the sparse matvec and reductions, rather than the SDF/legacy QP code.

### 4.5 Fixed-prior IEKF acceptance and mask-only points

The existing `update_points` code accepts per-point RGB-D iterates only when `nrows>0` under a chi² policy; that is **insufficient** for the new image measurement. A hidden point with `nrows==0` may have a nonzero rendered-mask image innovation/Jacobian and must be eligible for an update.

For `rendered_mask_measurement=on`:

- Keep the **original** RGB-D validity/reason codes, row innovation checks and RGB-D-only `nis`/`4*nrows` definition. Mask pixels are NOT RGB-D degrees of freedom.
- Implement **global coupled IEKF** (not per-point independent accept/revert) because changing one point can change the pseudomask and residual of other points at the same pixel. A per-point fallback applied after a coupled solve can invalidate the image model.
- Fixed prior and fixed measured frame for all linearizations; re-evaluate `M_hat(X)` and admitted RGB-D measurements at each iterate.
- Use a line search / trust strategy based on actual joint negative log-likelihood + fixed prior to ensure well-defined nonlinear iteration; do not silently allow rising objective or accept a solver that fails.
- If geometry rows are uninformative or all residuals zero, reduce to the existing unchanged RGB-D path. Any mask-only corrected point has its own explicit output status; `observed` and `nrows` continue meaning **direct RGB-D observed**.
- Do not interpret old chi-square RGB-D `last_consistent_iterate` thresholds as a goodness-of-fit test for binary mask rows. For `on`, track RGB-D consistency separately and joint objective separately.

## 5. Covariance handling: be accurate about what is exact

The **state MAP increment** in §4 follows the linearized sparse joint Gaussian-surrogate image model. **The corresponding full posterior covariance is in general NOT block diagonal**, because image pixels are shared by multiple points.

```text
A_joint = blockdiag(S_i^-1 + I_rgbd,i) + J^T R_mask^-1 J
P_xx,joint ~= A_joint^-1          # joint Gaussian/Laplace covariance; generally dense inverse
```

Do NOT claim that merely using `A_diag,i^-1` produces exact marginal covariances. That is a **conditional** block approximation and can be overconfident.

Required implementation/reporting:

1. Keep all prior full `6×6` point blocks and their `P_vx` cross-covariance. Recover updated velocity from the exact conditional relationship in §4.3.
2. Implement a **named, explicitly approximate** production marginal covariance policy (e.g. `pixel_marginal_adf`) for feasible memory/performance. One conservative candidate is to marginalize uncertainty from OTHER pixel contributors into each point's effective scalar mask noise for marginal covariance estimation:

   `R_eff[p,i] = R_mask[p] + SUM_{j≠i} J[p,j] S_j J[p,j]^T`,

   `S_i^+ approx = (S_i^-1 + I_rgbd,i + SUM_p J[p,i]^T R_eff[p,i]^-1 J[p,i])^-1`.

   This remains an **assumed-density approximation**, not an exact joint posterior; masks in multiple cameras may be correlated, and relinearization effects are ignored. Explicitly test whether it is over/under-confident, and avoid any claim of rigorously conservative bounds without proof. Alternatives may be substituted only with a documented derivation and tests.
3. Given chosen `S_i^+`, reconstruct each 6×6 block as:

   ```text
   P_i^+ = [ S_i^+              S_i^+ L_i^T ]
           [ L_i S_i^+  C_i + L_i S_i^+ L_i^T ]
   ```

   Check symmetry, SPD, Schur complement, finite values, and numerical scales; do not repair systemic errors by clamping eigenvalues to arbitrary floors.
4. On a **small independently solvable fixture**, explicitly assemble the exact sparse/dense joint Gaussian normal matrix, compute the true diagonal `3×3` blocks of `A_joint^-1`, and compare to the production covariance approximation. Report relative variance and NEES/consistency when GT is available. Include a two-point/one-pixel fixture demonstrating nonzero off-diagonal covariance.
5. Preserve existing correct RGB-D-only Joseph/information covariance parity when mask mode is off. Keep all versions of covariance assumptions in the run manifest.
6. The **binary observation plus anti-aliased coverage Gaussian surrogate** is an approximate image sensor model. Neither the covariance above nor the fitted mean should be called exact Bernoulli-posterior moments.

**Do not reintroduce `truncated_position_moments.py`, EP, hard-equivalent covariance projection, or expensive multi-halfspace moment integrals into this path.**

## 6. Mask measurement covariance / evidence provenance

A binary measurement has no automatically justified isotropic Gaussian variance. The image IEKF must use a **declared** positive `R_mask` for its anti-aliased occupancy measurement; it cannot select an arbitrary huge confidence to immobilize object points.

- Support `mask_image_variance` by camera/frame (and optionally pixel); store units `(white-intensity [0,1])^2` and provenance.
- Prefer actual segmentation uncertainty/calibration or pixel rasterization uncertainty provided with the acquisition. For synthetic exact masks, document that the Gaussian-surrogate noise/anti-aliasing is a modelling choice, not a measured sensor variance.
- If no defensible variance exists, keep configuration clearly labelled `UNCALIBRATED` and require the user to select/declare it. Do not claim calibration from the same RGB-D residual statistics or from GT joint displacement.
- Treat binary mask as a separate measurement channel in diagnostics; never add the mask pixel count to RGB-D `nrows`, degrees of freedom, or RGB-D NIS.
- Mask pixels derived from RGB segmentation are statistically correlated with RGB; an explicitly block-diagonal `R` is a composite-likelihood approximation. State this in manifests and reports.

## 7. Completely remove the old terribly slow hard-geometry implementation

This is a REPLACEMENT, not a coexistence of two production geometry systems.

After inspecting dependency references and preserving a snapshot:

- Delete the hard-QP/active-set enumeration path and any `reduced_qp`, `HARD_INFEASIBLE`, KKT restoration or candidate enumeration loops from the F1/B1 `update_points` execution.
- Delete/bypass `estimator/constrained_point_update.py` from production imports and tracked source hashes; remove the file if nothing else uses it.
- Delete/bypass `estimator/truncated_position_moments.py` EP/200-sweep posterior-moment path from production; remove the file if nothing else uses it.
- Remove the hard visual-hull/depth-free-space **inference path** in `estimator/visual_hull_constraints.py`; remove the module if no remaining diagnostic/reference depends on it. **Do not remove the original pre-existing RGB-D depth visibility rules**.
- Remove obsolete local `--geometry-constraints hard|diagnostic`, `--geometry-posterior-mean`, `--geometry-depth-mode`, `geometric_constraints` policy, `geometry_*` memory maps/traces/statuses and old hard-geometry debug UI **only after** migrating required diagnostics to the new renderer image measurement. If the user passes an old hard-mode flag, raise a clear error describing its removal; **never silently alias `hard` to the new image measurement**.
- Migrate/replace the README section and `reports/visual_hull_constrained_iekf.md` (if present) so they no longer present the hard EP path as the current algorithm. Keep any prior performance postmortem archived distinctly if useful.
- Replace obsolete hard-QP/EP test suites with tests for the **new required image measurement**. Retain general SPD/finite-difference test utilities and regression examples that are still applicable; do not delete unrelated healthy tests.
- Preserve the RGB-D-only `--rendered-mask-measurement off` mode for exact A/B comparison and source hash/resume safety.

## 8. Explicit CLI and mode contract

Proposed names (adapt exact names to existing project conventions; never collide silently with old flags):

```text
--rendered-mask-measurement off|diagnostic|on
--mask-image-noise-config PATH     # declared/calibrated mask-intensity variance
```

Policy object in `configs/projective_rgbd.json`:

```json
"rendered_mask_measurement": {
  "mode": "off",
  "render_white_points": true,
  "include_occluded_points": true,
  "foreground_as_measurement": "original_binary_image",
  "measurement_rows": "observed_background_only",
  "predicted_linearization": "bilinear_pixel_coverage_union",
  "binary_debug_renderer": "nearest_pixel_white_point_or",
  "joint_image_solver": "matrix_free_pcg_3d",
  "covariance_model": "pixel_marginal_adf_approximation",
  "measurement_noise_provenance": "required_when_on"
}
```

The mask image variance belongs in the dedicated noise configuration or equivalent camera/frame metadata, NOT as an arbitrary tunable `mask_loss_weight`.

`off` must match the legacy RGB-D path including trajectories, covariance, observation codes and no extra geometry computation. `diagnostic` renders real measured binary mask / predicted white-point pseudomask and reports image innovation but **never updates state or covariance**. `on` assimilates image residual into the SAME IEKF each frame. Select the default after validated A/B: start with `off` for development; do not declare `on` default until validation and runtime goals are met.

## 9. Diagnostics that MUST be delivered

Per selected frame/camera:

- Original measured foreground binary `M_obs` (white foreground, black background).
- Actual hard binary pseudomask `B` rendered from ALL points in white (include occluded).
- Red overlay `B & (~M_obs)`; **this is the exact semantic error requested**.
- Side-by-side anti-aliased predicted image used by IEKF, one-sided selected residual image `-(1-M_obs)*m_hat`, and accumulated pixel coverage / number of contributor points.
- Counts: points projected (frontmost / occluded classification for comparison only), projected inside FOV, background pixels touched, hard outside-white pixel count, total contributing point-pixel pairs, number of points with mask Jacobian rows, mask-only updated points, count with no usable image gradient, and number of overlapping contributors per pixel.
- Mask-specific `||r||`, `||J^T R^-1 r||`, linearization error, PCG iterations/relative residual, prior/RGB-D/mask objective components, acceptance and convergence.
- Per traced point: ID, `x^-,v^-,P^-`, each camera's projected UV, pixels/coverage weights, observed mask pixel values, actual image innovation values, contribution to `J^T R^-1 r`, mask-induced `Δx,Δv`, final state/covariance, RGB-D row validity (without conflation), and whether other points shared contributing pixels.
- Trace and compare named known-runaway points `storage_45135:171031` and `USB_100109:81029` when present in the actual point catalogue. Preserve the dataset/subsample ID mapping; do not assume those IDs exist under every subsampling factor.
- Save image diagnostics only for explicitly selected debug frames; **do not** dump full `C×H×W` images for every frame by default. Numeric summaries remain lightweight.

## 10. Non-negotiable tests

### A. Rasterization and measurement semantics

- One point projected into one background pixel → hard pseudomask white, `y_obs=0`, `y_hat=1`, exact `r=-1`.
- White/white → mask measurement row excluded. Black/black → zero residual.
- Two points on the same projected pixel → hard pseudomask still WHITE ONCE, not intensity two. Anti-aliased union remains in `[0,1]`.
- Occluded point behind another in the same camera → still contributes to mask renderer even if RGB-D is z-buffer rejected.
- Missing foreground white pixel (where no point projects) → NO mask correction.
- Outside-image or behind-camera point → not treated as observed black; other cameras still participate.
- Masks are actual data images, not fabricated/predicted sensor masks. No GT used in inference.

### B. Jacobian numerical oracle

On small non-grid-aligned configurations, compare analytic `∂m_hat/∂x_i` with symmetric finite differences of the full differentiable image renderer, including overlapping contributors. Test gradient on actual **pixel rows**, not a signed-distance function. Check camera Jacobians against independent finite differences. Explicitly check saturation and coincident points for finite values and deterministic subgradients.

### C. Exact sparse joint normal-equation oracle

For 2–10 points and 1–3 cameras, build a completely independent dense image-by-position Jacobian and normal system; compare matrix-free `Jv`, `Jᵀw`, `A(v)`, `b`, PCG `δX`, and objective. Test an overlapping-pixel case where off-diagonal terms are nonzero. Fail if implementation silently drops cross-point terms. Verify recovered `Δv_i = P_vx S^-1 Δx_i` against a full dense 6N IEKF calculation.

### D. Covariance

Verify SPD and symmetric `P_i^+` under forward/backward signed `dt`, all camera coverage patterns and overlap. Compare `3×3` approximate block covariances with exact diagonal blocks of the dense small joint inverse and quantify differences. Explicitly label approximation. Preserve original 6D Joseph/information parity in `off` mode.

### E. IEKF acceptance and observability

- Mask-only correction with **zero direct RGB-D rows** must be allowed when image measurement gives a valid gradient; `observed` remains false for direct RGB-D semantics.
- If the point is inside the measured mask and creates no background pixel coverage, there is no mask correction.
- **A point far outside in uniform black** may have a nonzero binary residual and zero useful image gradient. Verify and REPORT this known limitation rather than forcing an arbitrary fix or making a false observability claim.
- Off-mode parity for all existing arrays, output file paths, and results. Diagnostic-mode parity for state/covariance with additional report only.
- Fixed-prior iteration does not count the same mask pixels as independent observations. Relinearize and validate actual combined objective, not only step size.
- No loss of source/camera ID ordering or calibration convention (including the synthetic half-pixel fix).

### F. Realistic GPU/whole-pipeline regression

Run initial smoke tests on CPU and on GPU (if available), then both directions for **storage_45135, USB_100109, stapler_103111**, with masks from actual measured scenes, plus the existing ray-cast synthetic tests. Use the same source, calibrated intrinsics, time stamps, RGB-D noise and subsampled endpoint IDs for A/B comparisons. Report absolute numerical differences, not only pretty images.

Compare `off`, `diagnostic`, `on` for F1/B1. Then run downstream coherence → F2/B2 → joint → support without extra modifications; report failures transparently and do NOT claim the mask channel solved the existing `UNIDENTIFIABLE` joint issue without evidence. Re-run GT evaluation separately, with GT forbidden from inference.

## 11. Runtime, profiling and minimum acceptance

- First time the existing `off` F1 on storage and record wall/frame, GPU peak memory and IEKF iterations; preserve earlier reference (~1.8s/frame) only as historical comparison, not current exact timing.
- Profile separately: source frame transfer, white-point projection, hard binary rasterizer (diagnostics only), differentiable coverage/contributor compaction, image residual/Jacobian preparation, `Jv`/`Jᵀw` matvecs, PCG, RGB-D rows, posterior covariance blocks, and debug I/O.
- Show `# active point-pixel contributors`, `# touched background pixels`, `# overlapping pixels`, `# joint unknowns`, PCG iteration count/residual per IEKF iteration, memory peak and wall time.
- The performance requirement is **NOT** to replace 30–64s hard mode with another equally slow image-mode algorithm. Exploit `O(NC*4)` sparse projected contributions and matrix-free 3D linear algebra; do not perform an image-sized dense solve, full 6N dense factorization, exhaustive active sets, 200-sweep EP or CPU per-pixel loops.
- Benchmark at 30k, 100k and full point count under the same camera/frame settings, with and without overlap. Explain any superlinear cost from pixel overlap, sorting or PCG convergence. Check kernel launch and host synchronization cost.
- Do not fake an unachieved speed target, skip difficult points to improve time, or silently approximate the joint mean by independent point solves. A clearly labelled covariance approximation is permitted and must be tested.

## 12. Required final Codex report

Produce `reports/rendered_binary_mask_iekf.md` containing:

1. Final code diff/changed files; exact local git commit and source-hash manifest.
2. A one-paragraph confirmation that `y_obs` is **the measured binary next-frame mask**, `y_hat` is **the rendered white-point pseudomask**, and `r=y_obs-y_hat` is sampled **only where observed mask is black**.
3. The precise hard binary renderer and the differentiable pixel-coverage renderer, each with oracle tests, and their known mismatch/limitations.
4. Full sparse Jacobian/information equations, whether joint overlap coupling is retained and how GPU reductions remain correct.
5. Every removed hard-QP/EP import, CLI flag, output, and dead module; proof old hard solver is no longer used.
6. Posterior covariance contract (exact/approximate, errors vs small dense oracle, SPD checks); no false covariance exactness claims.
7. Off/diagnostic parity checks, on-mode numerical traces, RGB-D accuracy and GT evaluation on named scenes.
8. Frame timing, GPU utilization, memory and PCG convergence; comparative slowdown/speedup against both RGB-D `off` and former hard-QP/EP mode.
9. Explicit limitations including points with no mask-image gradient, shared pixel covariance and RGB/mask noise correlation.
10. Uncompleted steps and failing tests. Do not declare implementation finished until the full requested tests pass.

## 13. Final instruction

**Implement this exact measured-image innovation, not a differently motivated silhouette model.** The target is the existing 6D point-only forward/backward RGB-D IEKF augmented *organically* with a second image-sensor layer: the actual observed binary masks against the transported white-point rendered binary pseudomasks. Correct the state by the sparse renderer Jacobian and the same fixed-prior IEKF equations. **Remove the old terribly slow hard-geometry KKT/EP pipeline** after preserving a source snapshot and proving the new path and `off` baseline work. Keep independent point dynamics, admit occluded points into mask rendering, handle shared pixel couplings honestly, and report cases where a binary-only image residual is locally unobservable instead of silently inventing extra forces.
