# CODEX IMPLEMENTATION CONTRACT — Multiview Visual-Hull / Depth-Free-Space Constrained IEKF

**Project:** `SamueleMara/RGBDiekf`  
**Code baseline inspected:** `main` at `9a9f16f44165bb757739a5253321d910f62e2c1f` — *Drift Debug Update* (2026-10-28).  
**Target:** F1/B1 point-only, independent 6D RGB-D IEKF; retain the entire existing F2/B2 → joint → support chain.  
**Authority:** This specification deliberately supersedes prior proposals involving rigid coupling, velocity clipping, new outlier gates, or a Gaussian-splat renderer.

## 0. Nonnegotiable behavior

Implement **measured multiview silhouette feasibility** and **measured RGB-D free-space feasibility** as constraints on each point's own 6D IEKF state. They are derived from actual calibrated camera masks/depth, not tuned spatial/speed cutoffs. The estimator must continue to transport all points, visible or hidden.

1. **State remains independent per point:** `s_i = [x_i^W (m), v_i^W (m/s)]`, `P_i` a full SPD 6x6. No shared rigidity, no static/dynamic labels, no K-part prior, no supernode/group motion, no nearest-neighbor velocity transfer.
2. **Original dynamical model stays:** `F(dt)`, signed-time `Q(dt)` and `predict()` stay mathematically unchanged; backward processing uses the actual signed time interval.
3. **Masks are hard foreground data:** only the measured binary masks, including across *all applicable cameras*; no predicted/semantic masks, background residuals, fabricated 3D bounds or speed caps.
4. **Hidden points remain eligible for silhouette/depth feasibility even if no RGB-D measurement is visible.** `SELF_OCCLUDED`, `DUPLICATE_PIXEL_SUPPRESSED`, an RGB-D row rejection, and a missing direct RGB-D row MUST NOT disable geometric constraint evaluation.
5. **Silhouette and depth free space are distinct from RGB-D surface/color equality residuals.** Never force a hidden point onto the measured front surface. The inequality is `Z_point >= D_measured` for an ideal nearest-surface depth; a point may lie arbitrarily behind it if silhouettes and the other cameras allow it.
6. **No new gates / hardcoded physical tolerances:** no `max_velocity`, `max_displacement`, hand-picked 3D bounding volume, heuristic signed-distance penalty weights, fixed geometric slack, or per-scene tuned thresholds. Only solver numerical tolerances justified by precision and errors belonging to declared sensor/calibration uncertainty are permitted. Preserve *existing* RGB-D policies and their configurable legacy modes for controlled A/B testing; do not secretly change them in this task.
7. **Do not claim that masks reveal unknown hidden motion.** A point strictly inside all visual-hull constraints can remain unobservable. Its posterior should reflect this, not invent a velocity.
8. **No unannounced output/schema changes or silent failure.** Keep previous runs reproducible with `--geometry-constraints off` and identical source/noise manifests. New geometric influence must have its own status and diagnostics.
9. **No arbitrary covariance reset or equality-style zero-variance constraint.** Define and validate the posterior uncertainty model (§6) before enabling production constrained mode.
10. **No ground truth in inference.** GT may be used only in separate evaluators and regression tests.

## 1. Current implementation — exact insertion points

Line numbers below refer to commit `9a9f16f`; use symbol names and inspect diffs if the worktree has changed.

| File / location | Currently does | Required change |
|---|---|---|
| `estimator/projective_sensor.py:54–74`, `MeasuredFrame.__init__` | Reads per-camera `[C,H,W]` binary `mask`, `raw_valid`, metric `depth` and RGB | Add lazy/cacheable per-camera foreground **signed-distance field** and bilinear derivative support; leave existing RGB-D sampling unchanged. |
| `estimator/projective_sensor.py:89–132`, `sample` | Bilinear foreground RGB-D sampling, all-four-pixel validity and finite placeholders | Reuse its *depth* interpolation/gradient semantics when geometrically applicable, **without** reusing its direct-observation validity decisions or candidate list. New independent depth-constraint sampler. |
| `estimator/projective_camera.py:62–90`, `project`, `project_rows` | Calibrated `uv,Z,Ju,Jz` | Reuse exactly, including world-to-camera coordinates, optical-Z convention and analytic Jacobians. |
| `estimator/projective_visibility.py:20–58`, `visibility`; `:67–82`, `apply_frozen_assignment` | Nearest-depth frontmost pixel, duplicate suppression, frozen ownership | Keep for **direct** RGB-D measurement only. Never use this to decide whether a point contributes geometric silhouette/depth constraints. |
| `estimator/projective_terms.py:61–94`, `build_terms` | Direct RGB-D `r,H,R`, legacy/new consistency policies | Preserve. Reuse the derivative `Jz - grad_depth @ Ju` only inside the new geometric module, with the correct sign. Do not mix inequalities into the direct RGB-D row count or NIS. |
| `estimator/point_kalman_cuda.py:75–78`, `predict` | 6D transition and covariance | No changes to `F,Q` equations. |
| `estimator/point_kalman_cuda.py:164–225`, `PointMeasurement` | Per-point independent direct RGB-D contributions, `nrows` | Keep. Compute geometric terms for **all points independently of `nrows`** via a separate module. |
| `estimator/point_kalman_cuda.py:228–325`, `update_points` | Unconstrained 6D IEKF iterations, existing acceptance, Gaussian posterior covariance | Add opt-in constrained IEKF/SQP solve, nonlinear feasibility checks, separate geometry-conditioned covariance, geometry-only updates. Retain old code path bitwise/near-bitwise unchanged when disabled. |
| `estimator/free_projective_run.py:48–130`, `run_free` | Initializes independent 6D states; F1/B1 processing; writes traces | Prepare geometry per frame, pass into `update_points`; save geometric status and diagnostics; retain original point IDs and chronology. |
| `estimator/free_projective_run.py:136–156`, `_store`; `:156–235`, `_frame_numeric` | Per-point status and frame summaries | Add separate geometric fields. **Do not** overload RGB-D `observed`, `nrows`, `nis`, or `camera_reason`. |
| `configs/projective_rgbd.json:1–25` | Existing RGB-D policies, iteration budgets | Add a clearly namespaced `geometric_constraints` config with explicit mode, depth semantics, covariance approximation/reporting; do not retune existing RGB-D policies. |
| `train.py:36–46`, `STAGE_SOURCES` | Source hash/resume safety | Include new files and geometry config in `free` hash. |
| `train.py:56–105`, arguments; `:170–230`, stage runner | CLI, manifests, calibration, F1/B1 calls | Expose `--geometry-constraints {off,diagnostic,hard}` (no new physical thresholds), persist exact mode and version; thread through to F1/B1 without affecting other stages. |
| `debug/projective_debug.py` and `debug/assets/` | HTML / numeric exports | Add geometry fields to numerical traces and debug presentation if convenient; NEVER substitute visualization for the actual numerical trace. |
| `tests/test_point_kalman_cuda.py`, `tests/test_projective_measurement.py`, `tests/test_four_passes.py` | Existing unit/end-to-end tests | Keep old tests, introduce tests for exact geometric constraints, Jacobians, covariance, F/B behavior, and baseline parity. |

**New modules (names may be adjusted but must remain conceptually separate):**
- `estimator/visual_hull_constraints.py`: SDF construction, bilinear signed-distance sampling, depth lower-envelope sampling, per-camera constraints and Jacobians.
- `estimator/constrained_point_update.py`: 6D IEKF quadratic subproblem with 3D active-set reduction, feasibility/globalization, KKT diagnostics.
- `estimator/truncated_position_moments.py`: mathematically defined inequality-conditioned posterior moments and SPD checks.
- `tests/test_visual_hull_constraints.py`, `tests/test_constrained_point_update.py`, `tests/test_truncated_position_moments.py`.

## 2. Formal geometry — exact definitions

For point `i`, chronological frame `k`, camera `c`:

- World position `x_i,k`, world velocity `v_i,k`; camera projection `u = pi_c(x)` and optical depth `Z_c(x)`.
- Measured binary foreground `M_c,k(u,v) ∈ {0,1}`; measured metric first-return foreground depth `D_c,k(u,v)` where finite and valid.
- Foreground signed Euclidean distance `d_c,k(u,v)`, **positive inside** mask and negative outside. Define the signed raster convention exactly and unit-test it at foreground/background pixel centers. Recommended explicit convention: `d = EDT(mask) - EDT(~mask)` in **pixel units**, with a documented zero crossing under bilinear interpolation between pixel centers. Do not apply arbitrary erosion/dilation.

**Silhouette inequality:**

    gM_i,c,k(s) = d_c,k(pi_c(x)) >= 0

Its Jacobian is

    GM_i,c,k = [ grad_uv(d)^T * Ju_c(x), 0_(1x3) ]   (shape 1x6)

where `Ju = ∂pi_c(x)/∂x` (2x3).

**Depth free-space inequality (ideal first visible surface):**

    gD_i,c,k(s) = Z_c(x) - D_c,k(pi_c(x)) >= 0

and

    GD_i,c,k = [ Jz_c(x) - grad_uv(D)^T * Ju_c(x), 0_(1x3) ]

where `Jz = ∂Z/∂x` (1x3). This is the negative of the old `Hx` depth residual derivative because `projective_terms` uses `h_D = D-Z`, while the new feasible inequality is `Z-D >= 0`.

**Applicability must be explicit and independently reported:**

- Silhouette: applicable only when the projected point is in front of the camera and in the calibrated image footprint, and the mask for the **same object** exists. At the camera's image border, decide applicability using actual valid interpolation support; an out-of-FOV projection supplies *no evidence*, not an outside-mask penalty.
- Depth: applicable only if the same pixel has valid **foreground** depth and a finite, same-surface interpolation support (reuse the existing physical sample support contract). No depth returned => **no depth inequality**, but the silhouette constraint can still apply. An occluded 3D point remains eligible.
- Background and external occluders: never treat background depth as object depth. If masks do not represent the complete object silhouette or there are unmodelled external occluders, document that exact visual-hull feasibility is not established by those acquisitions; fail the hard-mode preflight if the dataset is known incompatible rather than silently making up geometry.
- The visual hull is intersected over **applicable** camera cones only. Camera coverage must be recorded for each point; cameras outside FOV do not assert negative existence. With inadequate coverage the feasible set can be unbounded—do not invent an enclosing box.
- All constraint terms use current measured frame `k`, not its anchor frame. Preserve the original F1/B1 camera ordering and timestamp semantics.
- Hard depth inequality is exact only in the ideal metric-depth model. For noisy real sensor depth, either (a) explicitly declare this a *hard idealized geometric assumption* and report conflicts, or (b) use a one-sided **censored measurement likelihood derived from the calibrated depth-noise law** (e.g. `-log Φ(gD/sigma_D)`), whose variance comes from `SensorNoise` and calibration uncertainty, not an arbitrary slack. Do not silently use hard inequalities as exact measurements when depth noise is acknowledged. Implement `depth_mode: off | hard_ideal | censored_noise`, document semantics, and make both branches independently testable. `hard_ideal` matches the ideal theoretical formulation.

**Important:** a hidden point may have `gM>=0`, `gD>0` but no direct RGB-D likelihood. That is a completely legitimate state. Never force `gD=0` for an occluded point or share its velocity with another point.

## 3. State prediction: unchanged

For each point, with `dt = timestamp[k]-timestamp[prev]` (signed for B1):

    s- = F(dt) s+_(prev)
    P- = F P+_(prev) F^T + Q(dt)
    F = [[I3, dt I3], [0, I3]]
    Q = [[|dt|^3/3 Qc, sign(dt) |dt|^2/2 Qc],
         [sign(dt) |dt|^2/2 Qc, |dt| Qc]]

`Q`, `P0`, calibrated `R`, velocity prior and existing RGB-D policies remain identical across off/diagnostic/hard A/B runs. Geometry should be the only changed experimental variable.

## 4. The constrained IEKF is a local SQP, not a penalty hack

At IEKF iterate `s_j`, the **existing** direct RGB-D information accumulator produces:

    Lambda = (P-)^-1
    A_j = Lambda + Σ_(direct valid rows) H^T R^-1 H
    b_j = -Lambda (s_j - s-) + Σ H^T R^-1 r

Do not reclassify hidden points as RGB-D observed merely because they supply geometry constraints. Build all applicable geometric inequalities independently:

    g_j = [gM_1,...,gM_C,gD_1,...,gD_C]^T    (only applicable entries)
    G_j = stacked [GM; GD]                      (m x 6)

Linearized geometric feasibility:

    G_j delta + g_j >= 0

At **every** relevant iterate, solve the strictly convex small QP:

    delta* = argmin_delta  0.5 delta^T A_j delta - b_j^T delta
             subject to   G_j delta + g_j >= 0

For a point with zero direct RGB-D rows, `A_j=Lambda`, `b_j=-Lambda(s_j-s-)` and the QP still exists: this is the required **geometry-only correction** of an occluded point. For a point without applicable geometry or direct RGB-D observations, the solution remains exactly the normal prediction.

KKT conditions for a feasible QP:

    A_j delta - b_j - G_j^T lambda = 0
    G_j delta + g_j >= 0,   lambda >= 0
    lambda ⊙ (G_j delta + g_j) = 0.

Record primal feasibility, dual feasibility, stationarity and complementarity. A multiplier `lambda` belongs to a specific camera and to a specific mask/depth inequality, so it identifies what physically constrained a correction.

### 4.1 Efficient **exact local** 3D reduction (do this; do not allocate a dense N-point covariance)

With `S_j=A_j^-1` (6x6) and unconstrained candidate `s_u=s_j+S_j b_j`, eliminate velocity analytically, since all current-frame geometric inequalities depend on `x` only.

Let `Sxx=S_j[:3,:3]`, `Svx=S_j[3:,:3]`, `x_u=s_u[:3]`, `v_u=s_u[3:]`. The equivalent position-only QP is

    x_c = argmin_x 0.5 (x-x_u)^T Sxx^-1 (x-x_u)
          subject to g_j + Gx_j (x-x_j) >= 0.

Then recover **the exact 6D constrained QP solution**:

    v_c = v_u + Svx Sxx^-1 (x_c-x_u)
    s_c = [x_c, v_c].

This preserves velocity correction via the actual position–velocity covariance, without any speed cap. Verify the 3D reduction numerically against a standalone 6D QP oracle on random SPD systems, including nonzero `Pxv`.

Implement a deterministic batched CUDA/Torch active-set or interior-point solver **on the per-point 3D systems**; avoid Python loops over every point, prohibit whole-cloud dense covariance and avoid CPU<->GPU transfers each iteration. The number of applicable inequalities per point is at most `2*C` before eliminating inapplicable cameras. Only a small number can be linearly independent at a generic 3D optimum. Degenerate/near-parallel rows need rank-revealing factorization and unambiguous failure diagnostics; never divide by an undocumented epsilon to hide singularities.

Numerical stopping tolerances are *numerical* (scaled by machine epsilon and matrix conditioning), not new physical distance/velocity limits. Expose solver residuals and convergence counts. On infeasible or not-solved subproblems, do not silently freeze, zero velocity, drop the constraint or publish the result as a feasible constrained update. Emit an explicit failure reason/trace; if production hard constraints cannot be enforced, fail the affected experiment visibly rather than claiming success.

### 4.2 Nonlinear feasibility and changing visibility

The silhouettes, projection and sampled depth are nonlinear/piecewise differentiable. A linearized QP is not proof that the updated point is actually feasible.

- Reevaluate the **actual original measured** SDF/depth inequalities at the candidate `s_j+delta*`.
- Reevaluate the RGB-D objective at its actual camera samples; track changes in row ownership, duplicate suppression and physical surface.
- Use a constrained-SQP globalization/feasibility restoration that minimizes the original MAP objective **subject to the actual geometry**, not by adding a hand-tuned penalty. The numerical line search must not silently weaken a constraint. On boundary non-differentiability, report the subgradient convention and verify with finite differences at smooth locations.
- Keep the original `update_acceptance` and `row_innovation_gate` switchable and unchanged for direct RGB-D rows. A fallback iterate must also satisfy the geometrical inequalities. Geometry feasibility is *not* equivalent to the existing RGB-D chi-square acceptance. If the current iterate is infeasible, solve for feasibility rather than declaring it valid on the basis of low photometric residual.
- Preserve the exact nonzero/zero `dt` conventions and chronology in B1.

If an exact hard intersection is empty because of calibration/mask conflict, **report hard infeasibility and its minimal conflicting camera constraints**; do not invent a slack or change the noise configuration. A separately selected calibrated `censored_noise` depth branch can represent real sensor uncertainty; it is not a silent fallback.

## 5. What the constraints can and cannot infer

- An **occluded, moving point** with an incorrect extrapolation that crosses a multiview silhouette boundary gets a position/velocity correction through `Pvx` (or through the smoothing transition coupling); it does not need RGB-D surface visibility.
- If the point remains strictly inside all hulls, the masks provide **no deterministic evidence about its internal velocity**. No proposed algorithm may hallucinate that velocity or claim the true trajectory is observable. It retains its dynamic prior uncertainty and can be constrained when observed again.
- An image pixel with a foreground mask but no valid depth supplies silhouette information, not a fictitious depth measurement.
- An out-of-FOV camera supplies no object-absence assertion.
- Do not add rigid-group propagation. This is independent geometric feasibility, not cross-point rigidity.

### 5.1 Velocity feasibility across frames; smoothing is a **separate explicit mode**

The current-frame C-IEKF already corrects velocity through position–velocity covariance. For offline motion discovery, the theoretical next-frame geometric constraint on the **previous** state is

    g_(k+1)( [x_k + dt v_k, v_k] ) >= 0
    G_(k->k+1) = [ a_(k+1)^T, dt*a_(k+1)^T ],

where `a_(k+1)^T = grad_uv d_(k+1)^T Ju_c(x_k+dt v_k)` (and similarly for depth), evaluated using actual measured frame `k+1`. This **explicitly** constrains velocity, rather than bounding its magnitude. In a causal forward pass, the constraint is only used after frame `k+1` is acquired; using it earlier is noncausal smoothing.

If a later optional temporal-smoothing mode is implemented, use a **separate**, independent-per-point, block-tridiagonal constrained trajectory MAP problem:

    min_(s_i,0:T)  0.5 ||s_i,0 - mu_i,0||^2_(P_i,0^-1)
                  + 0.5 sum_k ||s_i,k+1 - F_k s_i,k||^2_(Q_k^-1)
                  + 0.5 sum_(k,c in direct visible observations) ||r_i,c,k||^2_(R_i,c,k^-1)
    subject to       gM_i,c,k(s_i,k) >= 0,
                     gD_i,c,k(s_i,k) >= 0  (where applicable).

Do not call this an IEKF-only update; it is a batch/fixed-lag constrained smoother. Do not equate its state/covariance with the independent forward-only filter or merge distinct forward/backward canonical point identities. **This smoothing extension is not required for the initial C-IEKF implementation;** document the formula and retain a safe insertion point, but do not introduce it silently into F1/B1 or F2/B2.

## 6. Posterior uncertainty: **do not treat active inequalities as equality measurements**

The ordinary IEKF covariance `S=A^-1` is the local *unconstrained* Gaussian posterior approximation. It is not the covariance of the state conditioned on a hard inequality. Do **not** (a) leave this issue undocumented, (b) collapse normal variance to zero as though `g=0` were an exact equality, or (c) reset `P` to the initial value after projection.

The required formal posterior semantics are **inequality-conditioned Gaussian moments** under the local affine constraints. This is an assumed-density approximation to the original nonlinear constrained posterior, explicitly labelled as such.

### 6.1 One affine constraint: analytical reference (mandatory)

Given local Gaussian `s ~ N(mu,S)` and affine constraint `a^T s >= b`, let

    sigma^2 = a^T S a
    alpha = (b-a^T mu)/sigma
    lambda = phi(alpha)/(1-Phi(alpha))

Then conditional moments are

    mu_c = mu + (S a / sigma) * lambda
    S_c = S - (S a a^T S / sigma^2) * lambda * (lambda-alpha).

Implement stable log-survival and Mills-ratio evaluation for extreme `alpha` (no NaN/Inf or catastrophic cancellation). The normal-direction covariance remains **nonzero** for finite data. Verify against numerical integration / Monte Carlo in tests, not as the production estimator.

### 6.2 Several simultaneous current-frame constraints: reduce moments to **3D position**

Constraints depend only on `x`. Denote the local *unconstrained* joint Gaussian blocks:

    mu = [mu_x, mu_v]
    S  = [[Sxx,Sxv],[Svx,Svv]],  B = Svx Sxx^-1.

Let `(mu_x^C, Sxx^C)` be the first two moments of the **3D Gaussian `N(mu_x,Sxx)` conditioned on the intersection of local affine position half-spaces**. Then reconstruct the full 6D moments **exactly from those 3D moments**:

    mu_v^C = mu_v + B (mu_x^C - mu_x)
    S_vx^C = B Sxx^C
    S_xv^C = (S_vx^C)^T
    S_vv^C = Svv - B Sxv + B Sxx^C B^T.

This is algebraically required by the conditional Gaussian law. Verify SPD/PSD, symmetry, cross-covariance and independence of camera constraint order.

Use a **deterministic, accuracy-controlled 3D truncated-normal moment algorithm** for multiple half-spaces; not a heuristic midpoint, hard boundary projection, or sequential one-constraint covariance overwrite claimed as exact. A low-dimensional integration/assumed-density approximation may be used only if its order dependence, integration error and covariance properties are documented and regression-tested against independent high-accuracy numerical references. Surface the mode/approximation in the manifest and debug trace. Do not call the approximation “exact covariance.”

**Important semantics:** The constrained MAP solution `s_MAP` (from §4) and the conditioned Gaussian **mean** `mu_C` are not generally identical. Never write the MAP mean together with a covariance of an unrelated distribution and silently claim they are its moments. Choose explicitly and document what is propagated:
- **Preferred probabilistic filter:** propagate the inequality-conditioned mean `mu_C` and moment covariance `S_C`, and save the constrained MAP as an auxiliary diagnostic. On local convex halfspaces the truncated mean remains in the linear feasible set; check the true nonlinear constraints before accepting it.
- If implementing a **MAP-trajectory mode**, record its distinct estimator semantics and its local covariance approximation. It must not claim to be a moment-matched Bayesian posterior.

A production hard-constrained mode is **not complete** until it has a coherent uncertainty update. If multi-constraint moments cannot be computed to a declared accuracy and memory budget, leave the mode diagnostic/experimental and report the blocker; do not hide it behind `P=A^-1`.

### 6.3 No active geometric information

With no applicable constraint, return the **identical** unconstrained IEKF `(s+,P+)` (modulo deterministic floating-point evaluation order), and do not spend time computing truncated moments. When geometry is only theoretically relevant in a far Gaussian tail, numerical integration must report its contribution/error; any computational shortcut must be disclosed as an approximation, not as a physical gate.

### 6.4 Calibrated noisy depth option

For noisy first-return depth `D = Z_front + eps`, where `eps ~ N(0,sigma_D^2)` and true occluded point obeys `Z_i >= Z_front`, use a **one-sided censored likelihood** rather than a hard `Z_i >= D` if chosen:

    p(D | x_i is behind nearest surface) ∝ Phi((Z_i-D)/sigma_D)
    E_depth_free = -log Phi(gD/sigma_D).

It is an uncertainty-driven likelihood, **not a gate or velocity cap**. Account for its gradient/Hessian in the MAP solve and define the resulting covariance approximation; do not mix this optional likelihood invisibly into the hard inequality set. If the physically correct noise/censoring model is unknown, run `hard_ideal` only as a declared idealized hypothesis and report failures.

## 7. Implementation sequence — in order; stop rather than inventing substitutes

### Stage A — geometry only, **no inference changes**

1. New `estimator/visual_hull_constraints.py` with:
   - `make_signed_distance(mask: [C,H,W] bool) -> [C,H,W] float` plus precise raster convention; CPU EDT during frame preparation is acceptable initially, then copy to GPU **once per frame** (avoid host-sync inside the inner IEKF).
   - `sample_sdf_and_gradient(sdf,uv,cam)` using the **exact analytic derivative of the bilinear SDF interpolant**; validity/FOV and finite placeholders.
   - `sample_depth_free_space(frame,uv,cam,noise,policy)` preserving foreground / raw-valid / same-surface semantics but NOT frontmost/duplicate suppression; return depth, gradient, depth variance, valid flag and reason.
   - `evaluate_constraints(s: [N,6], frame, rig, *, point_subset=None)` returning dense-or-packed `(point_id,camera_id,type,g,G,applicable)` and separate per-point summaries.
2. Add `geometry_constraints=diagnostic` to `train.py` / `free_projective_run.py`; run all points each frame but **never modify** the mean/covariance in this mode.
3. For traced points, export all camera inequalities at the predicted state and at the ordinary post-update state, including camera coordinates, FOV applicability, SDF values (px), raw/bilinear depth (m), Jacobians, and a boolean actual inequality violation.
4. Prove mode `off` remains identical in states/covariances/status to the baseline with equal config, precision, point selection and seed.

**Gate A must pass before Stage B:** analytic SDF/depth Jacobians match central finite differences at smooth samples; frame-by-frame measured mask hashes and camera geometry are unchanged; hidden points are included even if `nrows=0`.

### Stage B — deterministic *local QP oracle* on a handful of points

1. New `estimator/constrained_point_update.py`, implement the 3D reduced local QP from §4.1.
2. Verify it against a separate SciPy/high-precision **6D quadratic optimizer oracle** (test-only). Check primal/dual KKT, and velocity correction `Svx Sxx^-1 Δx` to numerical precision.
3. Probe point `171031` at storage forward frame 1 and point `81029` at USB forward frame 2 **using original point ID mapping and original noise policy**. Report each camera constraint before/after the original spike, whether it is violated and which multiplier would be active. Do not assume the visual hull catches an error if the point remains inside the hull.
4. Test a synthetic occluded point with all direct RGB-D rows absent: a true geometric violation must correct position **and velocity** through `Pvx`; an interior point with no evidence must retain its prediction.

**Gate B:** no inserted physical thresholds; exact QP/KKT equality against independent oracle; camera/constraint identities in trace.

### Stage C — constrained posterior moments

1. Implement one-halfspace exact formulas and 3D multi-halfspace moments per §6, with stable numerics.
2. Reconstruct 6D mean/cov from position moments, check all SPD/PSD/cross-covariance identities.
3. Compare single/multiple-constraint examples to independent integration references and convergence as numerical accuracy is tightened; do not rebrand a deterministic approximate method as an exact Bayesian posterior.
4. Profile memory/time for realistic `N` and 8 cameras **before** making this a production default.

**Gate C:** uncertainty contracts and per-frame resource budgets verified; no silent unconstrained covariance reuse when geometry actually changes the posterior.

### Stage D — integrate into F1/B1 and test downstream

1. In `PointMeasurement` leave direct RGB-D measurement calculation unchanged. In `update_points` optionally compute all geometrical terms for **every point**, not just `act` / `observed`, and solve constrained steps; reevaluate actual nonlinear geometry and direct RGB-D likelihood.
2. Distinguish **direct-RGB-D observation** from **geometry-only** posterior correction. Keep `observed` and `nrows` denoting only accepted direct image measurements. Add fields `geometry_applied`, `constraint_types`, `geometry_valid_cameras`, `geometry_violation_*`, `geom_active_count`, `geom_multiplier_*`, `geometry_solver_status`, `geometry_covariance_method`.
3. Use a **new** `geometry_update.npy` and `geometry_summary` fields; do not silently introduce a third value into `point_status.npy` until every downstream consumer has been checked. A geometry-only point remains `PREDICTION_ONLY` in the *direct RGB-D* status sense; its predicted state may be corrected by geometry and this fact is recorded separately.
4. Thread mode through `run_free` for both F1/B1. Retain anchor gauges, original point IDs, timestamps, canonical RGB, endpoint backprojection covariance, output names, downstream stage interfaces.
5. Do not automatically modify F2/B2/joint/support algorithms or reinterpret their Gaussian assumptions. Run them unchanged on the new free trajectories, and report whether constrained posterior covariance/motion-support changes affect them.
6. Register files in `STAGE_SOURCES['free']`, create structured manifest metadata, reject unsafe resume if geometry mode/source hashes differ.
7. **Only if** all gates/tests pass, turn the constrained mode on by an **explicit chosen configuration**. Keep `off` as an exact reproducible control. Do not silently switch production to a partial implementation.

## 8. Essential numerical debug files

Create a small, precisely scoped diagnostic artifact under each free pass, e.g. `free_forward/numeric/geometry_trace_frame_0001.json`, emitted only for `--debug-point-ids` and `--debug-frames` (avoid massive per-frame pixel dumps).

For **each traced point / camera / IEKF iteration** record:

- IDs and provenance: scene, commit hash, policy/noise hash, direction, chronological frame and processing step, selected-to-full point index, originating camera and pixel, measured RGB-D source hashes, timestamp `dt`.
- `s_minus,P_minus,s_j`, direct RGB-D `nrows`, `valid_cameras`, reason, original `r,H,R,NIS` and existing acceptance decision.
- Projection `u,v,Z`, `in_fov`, silhouette applicability, `signed_distance_px`, interpolated SDF pixel coordinates, local SDF gradient, `GM` (1x6), depth applicability, `D, gradD, noise variance, gD, GD`, underlying 2x2 pixel support and masks.
- `g` and `G` **before** and **after** the proposed correction; camera/type-specific constraint IDs; actual nonlinear feasibility (not only the linearized QP value).
- Local `A,b,Sxx,Svx`, unconstrained `delta`, constrained `delta`, prior/measurement/geometric correction components, `lambda`, and full primal/dual/KKT residuals.
- Nonlinear RGB-D residual objective before/after **on fixed common measurement rows** and on the newly actual visible row set, separately. A decreasing sum caused merely by disappearing rows is not proof of a correct state.
- Posterior `s_MAP`, conditioned `mu_c`, `P_c`, SPD eigenvalues, moment integration numerical error/convergence, geometry-only vs direct-measurement classification.
- Any infeasibility with the exact conflicting camera row IDs. **Never** substitute manually selected positions, fabricated depth, or GT during inference.

Global per-frame statistics: number of eligible mask/depth constraints, fraction with violations at `s-` and ordinary `s+`, observed vs geometry-only corrections, position and speed distributions **without clipping**, convergence/failure counts, covariance consistency metrics, runtime and peak GPU memory.

Keep the existing `debug/data/*.js` viewer export compatible; add optional geometry toggles if useful but prioritize numeric evidence.

## 9. Tests — mandatory independent references

Create the following tests using measured binary silhouettes generated independently from the estimator's predictions:

1. **SDF sign and units:** inside positive, exterior negative, zero-crossing convention documented, no background-as-depth, no mask erosion/dilation, 2x2 bilinear gradient matches finite difference.
2. **Perspective chain rule:** camera `Ju,Jz` and `GM,GD` agree with finite differences on multiple camera poses and finite image samples, including signed depth Jacobian direction.
3. **Occlusion independence:** two points on one ray, frontmost point is the only *direct RGB-D* sample; hidden point gets silhouette/depth feasibility and can be corrected independently.
4. **No invented motion:** hidden point strictly feasible and with zero direct rows stays prediction-only in an unconstrained-MAP sense; geometry must not invent a moving part or make normal covariance zero. The exact truncated-moment mean can shift slightly due to Gaussian tail conditioning; distinguish this analytically from deterministic MAP.
5. **Depth inequality:** visible surface `Z≈D`, hidden `Z>D` acceptable; point `Z<D` violates; raw-invalid/foreground-depth-missing supplies no depth factor; depth is never used as equality for occluded points.
6. **Multi-camera hull:** a point allowed by 1 camera but outside another in-FOV foreground mask is infeasible; outside another camera's FOV is **unknown**, not infeasible.
7. **Local QP:** independent 6D reference, 3D elimination, KKT, exact velocity correction, dependent/parallel constraints, infeasible intersections and zero RGB-D rows.
8. **Truncated covariance:** one halfspace exact normal moments; multiple halfspaces vs independent high-accuracy reference; symmetry, positive (semi)definiteness, stable tail/`logcdf`, correct x-v cross covariance.
9. **Backward time:** same geometric coordinate constraints at a chronological frame; signed `dt` preserved, no implicit reverse-velocity sign flip or endpoint gauge change.
10. **Off-mode parity:** original arrays `posterior_xyz.npy`, `velocity.npy`, `point_status.npy`, `valid_cameras.npy`, `camera_reason.npy` and covariance reproduce baseline with identical inputs; no new effect on direct RGB-D gates.
11. **Synthetic moving part:** actual moving points keep their motion; temporary hidden surfaces propagate and are constrained only where the geometry supplies information; no false stationary assumption.
12. **Synthetic fully static scene:** no artificial silhouette push deep inside the feasible hull; geometry does not force static labels, and no false joint should be introduced downstream.
13. **Real sequences:** USB/stapler/storage + available real laptop scene, **forward and backward**. Evaluate point `171031` (storage F1 frame 1) and `81029` (USB F1 frame 2) with exact original IDs. Report whether original failure state violates masks/depth, and whether corrected state changes its wrong RGB-D likelihood; do not claim improvement solely because maximum speed is smaller.
14. **Performance:** same camera count, full point population, 60 frames, GPU memory and latency. Never test only a tiny downsample and claim production feasibility.
15. **Full suite:** run existing unit tests and four-pass end-to-end tests under `--sensor-noise config` (baseline comparison), explicitly test measured noise where available, and compare downstream rigid/joint/support status. Do not modify existing pass criteria to conceal regressions.

Suggested *test-only* commands (adapt to locally installed env / actual scene roots):

```bash
python -m unittest tests.test_visual_hull_constraints tests.test_constrained_point_update tests.test_truncated_position_moments
python -m unittest tests.test_projective_measurement tests.test_point_kalman_cuda tests.test_four_passes
```

For a small F1 diagnostic, with the same unmodified source scene and original point selection:

```bash
python train.py --scene "$STORAGE_SCENE" --output "$OUTROOT/geometry_storage_diagnostic" \
  --direction forward --stop-after free --frame-limit 3 --device cuda \
  --pcd-subsample-factor 1 --covariance-precision float64 \
  --sensor-noise config --geometry-constraints diagnostic \
  --debug none --debug-frames 0,1,2 --debug-point-ids 171031 --no-evaluate --no-qa

python train.py --scene "$USB_SCENE" --output "$OUTROOT/geometry_usb_diagnostic" \
  --direction forward --stop-after free --frame-limit 4 --device cuda \
  --pcd-subsample-factor 1 --covariance-precision float64 \
  --sensor-noise config --geometry-constraints diagnostic \
  --debug none --debug-frames 0,1,2,3 --debug-point-ids 81029 --no-evaluate --no-qa
```

**Reproduction requirement:** These IDs come from an earlier run. Before claiming exact trajectory reproduction, verify the source dataset, camera order, canonical catalogue size/index mapping, endpoint subsampling method/seed, timestamps and `--noise-config` match the old run. When they do not, derive the matching point from canonical XYZ/source pixel and record the new ID; never assume two different catalogues use the same ID. For B1 use the full original sequence so the anchor remains at the actual final frame; truncating to a prefix changes the backward problem.

## 10. Deliverables and final review table

Codex must return:

- A clean, surgical diff. `off` path remains functional and reproduces old output. No unnecessary refactors of rigid/joint/coherence.
- Exact filenames, function signatures and module interfaces; shapes, physical units, F/B timestamp conventions; source hash/manifest updates.
- An actual constrained-SQP solution, an explicitly correct posterior covariance **semantics**, and independent mathematical tests. If one of these is unresolved, **say so** and do not declare `hard` complete.
- A per-point numeric trace proving mask/depth feasibility before/after on the known runaway examples, distinguishing mask constraints, depth constraints and direct RGB-D evidence.
- Comparative actual outcomes on static and moving populations, rather than only speed/displacement reductions.
- A compact results table:

| Gate | Check | Required evidence |
|---|---|---|
| A | Geometry oracle | SDF/gradient/depth valid, same frame, camera calibration and mask hashes |
| B | QP correctness | 3D/6D parity, full KKT residuals, independently computed x-v correction |
| C | Posterior covariance | Analytical single constraint, verified multiconstraint moments, SPD/PSD and accuracy |
| D | Integration | Hidden geometry-only updates, exact off-mode parity, signed backward time |
| E | Real examples | Storage `171031`, USB `81029` with actual g/G/camera rows and failure traces |
| F | Downstream + performance | F2/B2/joint/support preserved or explicit measured regression; full-scale GPU profiling |

**Implementation acceptance is scientific, not cosmetic:** Do not equate “no point flies away” with recovery of correct motion. The goal is to make each 6D state statistically and geometrically consistent with the *actual measured multiview RGB-D and binary foreground masks*, including when that state is hidden from direct RGB-D observation.
