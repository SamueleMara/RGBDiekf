# CODEX — RGBDiekf sparse image-gradient motion guidance and multiview part-motion tests

**Date:** 2026-11-02  
**Target repository:** `https://github.com/SamueleMara/RGBDiekf`  
**Verified public commit:** `9465cb109f22e4047d7132b3f9d2f22f3f5d633e`  
**Local working directory reported by Codex:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Priority:** Physically correct *image-plane RGB tangential guidance* + *depth surface-normal guidance*, fused across **every eligible camera**, with tests establishing whether it prevents dynamic-point velocity drift **without moving static material**.

**Status:** This file is an implementation specification for Codex, not completed code. Important: the user's `local-correlated` and `material-following` changes are **uncommitted** relative to the public baseline. **Inspect actual `git status`, local code, existing experiment reports and current CLI before editing. Do not checkout/reset/overwrite the working tree.** Modify surgically and preserve all existing modes and tests.

## 0. Problem and validated input baseline

The most recent supplied Blade run has:

- `run_manifest.json`: `observation_model=temporal-rgbd`, `temporal_mode=material-following`, `direction=both`, `frame_limit=null`, full cloud (`pcd_subsample_factor=1`), 8 cameras, 60 frames, acquisition metadata **59 Hz** (`dt=1/59 s`); reference policy `latest-valid`, lag=3; `material_resampling=none`, `pyramid_levels=0`, `temporal_warp_noise=ideal`, `rendered_mask_measurement=off`.
- `result.json`: forward 34,369 points and backward 38,622 points, both 60-frame runs completed; completion **does not mean scientifically accurate motion**. Their endpoint point sets differ, so don't directly align arrays by local index.
- `noise_calibration.json`: RGB sensor variance `[1.2815583749839803e-6]*3` (encoded RGB), depth sensor variance `4.736951571734001e-15 m^2`; cross-camera mismatch **not** included, additional warp covariance **zero**, explicitly **uncalibrated ideal-warp assumption**. Static-majority consecutive-frame MAD and quantization are conditional sensor estimates, *not* a validated temporal correspondence error model.
- `run_manifest.json`: initial velocity covariance `I_3 (m/s)^2`, process spectral density `0.01 I_3 m^2/s^3`. Do not change these as a shortcut to hide failed observations.
- User-observed issue: static points stable; moving points fluctuate/drift, sometimes acquire spurious high velocity and later run prediction-only. The explanation is an **unreliable direction of observation guidance and insufficient material correspondence**, not simply numerical covariance inversion.

**Objective:** Predict the **physical temporal change of RGB and metric optical depth caused by 3D surface motion**, using **2D image derivatives computed in pixel coordinates**, then map their constraints into a common **world-frame 3D velocity**. Use per-camera first-depth visibility. Diagnose and avoid false static-point motion from spatial gradients, occlusion, bias and repeated image measurements. Provide an optional **shared rigid-part twist** experiment to test whether reliable regions can guide textureless moving regions, without claiming unknown part membership is solved.

### Hard restrictions

1. Keep original calibrated cameras, raw measured RGB-D, foreground binary masks as hard constraints, and hard **first-depth** point ownership. No inverse-depth/background blending, no alpha compositing, no 3DGS, Gaussian splats, normal/surfel/mesh *state*, or new rendering model.
2. No GT values, GT part memberships, GT joint positions or GT velocities may enter deployable inference or change parameters or acceptance. GT may be used in **separate offline oracle/evaluation** processes and explicitly segregated outputs only.
3. Keep the original **independent point state** `s_i=[x_i,v_i]` (world coordinates, 6D) in F1/B1. An optional part-based motion state is a **distinct experimental stage** requiring an independently provided/validated non-GT association. No implicit all-point coupling or taking neighboring points as one rigid part.
4. No arbitrary new thresholds, gates, speed caps, damping, covariance inflation, zero noise substituted for measured data, pretrained flow, forced static/dynamic labels, manually assigned dynamic IDs, or hidden fallback. Keep the existing policies for controls; report their effect separately.
5. Preserve `legacy`, `simplified`, `local-correlated`, `full`, and `material-following` original behavior and outputs. Introduce a new **opt-in** mode, plus diagnostic-only facilities. Regression tests must still run.
6. Do not add the same RGB-D pixel evidence twice. The velocity-gradient differential form is closely related to the existing material-following residual. **Replace** the existing RGB-D likelihood in a selected test arm, or use gradient equations only as diagnostics. Never independently assimilate both on the same pixels without a proven joint noise/cross-covariance model.
7. Do not say “all eight cameras contribute” merely because eight are calibrated. Report **geometrically frontmost**, **material-eligible**, **usable derivative**, **admitted**, and **actually assimilated** per point/camera.

## 1. Inspect and map the REAL repository before coding

Inspect actual HEAD, modified/untracked files, `reports/temporal_local_correlated_validation.md`, `reports/temporal_dynamic_point_recovery.md`, relevant GT-only patch studies, current F1/B1 output shape and tests. Keep source hashes in the new report.

Known **committed code reference points** (verify local versions):

| File/function | Existing responsibility | Required treatment |
|---|---|---|
| `estimator/temporal_zbuffer.py:first_hit_pixel_owner`, `selected_owners` | All cameras' hard nearest-integer depth-sorted winning point per pixel | Reuse. Retain *all* point projections as competitors; do not render background points into a winning pixel. |
| `estimator/temporal_reference.py:ReferenceTable` | Same-camera `(point, frame, pixel, state)` reference and accepted lifecycle | Reuse; preserve source identifiers and reference noise. Audit lag=3 dropout/reacquisition separately. |
| `estimator/temporal_rgbd_terms.py:temporal_prediction`, `build_temporal_rows` | Temporal RGB-D residual with `H_t=[H_x,0]`, `H_ref`, per-camera rows | Reuse model/IDs; implement separate gradient adapter. Never add gradients as duplicate rows. |
| Current **uncommitted** `estimator/temporal_local_covariance.py`, `temporal_local_iekf.py`, `temporal_local_run.py`, and material-following modules (discover names) | Within-point joint state/reference/raw-pixel uncertainty | Reuse this as the statistical backend; **do not rebuild** global shared-pixel graph. |
| `estimator/temporal_simplified_iekf.py:information` | `index_add_` of RGB-D information across admitted camera rows per point | Control only; confirms camera fusion exists; not a new velocity supervision model. |
| `estimator/point_kalman_cuda.py:transition,process_noise,predict` | Existing 6D constant-velocity prediction and covariance | Reuse without changing F/Q or noise policy. |
| `estimator/projective_camera.py:CameraRig` | World-to-camera projection, `J_pi`, optical depth `J_Z`, calibrated extrinsics | Reuse; enforce row/column, camera handedness and units. |
| `estimator/projective_sensor.py:MeasuredFrame` and `scene/noise_calibration.py` | Mask-positive measured RGB-D and sensor-noise provenance | Reuse; include separate temporal derivative and correspondence discrepancy accounting. |
| `estimator/projective_coherence.py`, `estimator/motion_coherence_support.py`, `estimator/rigid_kalman_cuda.py`, `estimator/se3.py` | Existing downstream non-GT coherence/rigid stages | Audit their actual interfaces before optional part-guided mode. **Free F1 estimates are not automatically trustworthy membership evidence.** |
| `train.py`, `tools/run_temporal_rgbd_critical.py`, `utils/temporal_output.py`, `debug/temporal_debug.py` | CLI, run orchestration, journal, diagnostics | Opt-in dispatch, reproducibility and failure-safe output. Update source hash inventories. |

Before implementation write `reports/gradient_motion_code_map.md` with exact local functions/line references, current original material-following residual and covariance assumptions, and potential double-use of camera pixels.

## 2. Physical image-formation equations — no invented normal-only RGB forces

For a fixed calibrated camera `c`, world point `X(t)`, camera projection `u_c=pi_c(X)` and **metric optical depth** `Z_c(X)`:

\[
 \dot u_c=J_{\pi,c}(X)\,v,\qquad v=\dot X\in\mathbb R^3.
\]

### 2.1 RGB material constancy in *image coordinates*

At the *same visible material* and under locally valid appearance constancy:

\[
 I_c(\pi_c(X(t)),t)=I_{\rm material,c},
 \quad \partial_t I_c+\nabla_{u} I_c J_{\pi,c}\,v=0.
\]

For RGB channels stack `grad_u I` into a `3x2` matrix. Define RGB velocity row matrix and temporal RHS:

\[
 G_{I,c}=\nabla_{u} I_c J_{\pi,c}\in\mathbb R^{3\times3},
 \qquad b_{I,c}=-\partial_t I_c\in\mathbb R^3.
\]

**Important physical distinction:** large spatial `grad_u I` is *not* a motion observation: `b_I` must contain valid temporal change of the *same material* (or use finite-displacement material-following residual below). On a static surface under stable lighting, `b_I≈0`; mere texture must not move it. Also: a single scalar gradient constrains flow **along the gradient**; isophote-tangential motion remains unconstrained. Do not confuse “surface tangent” with “isophote tangent.”

### 2.2 Depth: normal component from image-space depth gradient

Material surface identity implies

\[
 D_c(\pi_c(X(t)),t)=Z_c(X(t)).
\]

Differentiate:

\[
 \partial_tD_c+(\nabla_u D_c J_{\pi,c}-J_{Z,c})v=0,
\]

or equivalently

\[
 \boxed{(J_{Z,c}-\nabla_uD_c J_{\pi,c})v=\partial_tD_c}. 
\]

`J_Z=dZ_c/dX = e_3^T R_cw` (fixed camera), `J_pi` world-frame perspective projection derivative. The row `a_c=J_Z-grad D J_pi` is parallel to the local visible surface normal **only where `D` is a valid differentiable single-valued surface**. The normal is a **derived diagnostic** from the current depth image and **not stored in the point state**. At depth breaks, silhouettes, disocclusion and wrong-surface comparisons, this equation is not an admissible velocity measurement.

### 2.3 Local tangent/normal coordinates (DIAGNOSTIC BASIS, not a new state)

At a valid depth pixel `u`, back-project `X_c(u)=D_c(u) K_c^{-1}[u_x,u_y,1]^T`. From image derivatives `X_u,X_v`, form `n_c ∝ X_u × X_v`, transform to world with `R_wc`, then choose stable orthonormal tangent columns `T=[t1,t2]` and `B=[T,n]`. Assert `B^T B≈I` and compare `n` with normalized depth row `a_c` (sign may differ; compare absolute cosine, retain signed equation).

Write `v=B*[alpha_1,alpha_2,beta]^T`. In this basis:

\[
 G_{I,c}B=\left[\nabla I_c J_{\pi,c}T\mid\nabla I_cJ_{\pi,c}n\right],
\]

\[
 (J_Z-\nabla D J_\pi)B\approx[0,0,\pm\|a_c\|].
\]

RGB can respond to *both* tangential and normal motion; depth primarily constrains normal motion. **Do not set RGB normal columns to zero or force depth tangent columns to zero numerically.** Report them and measure discrepancy to the ideal geometry. If normal estimation is ill-defined, use the full 3D world-frame row directly; never manufacture a tangent basis.

### 2.4 Fixed image pixel vs material transport; differential vs finite displacement

A naive finite difference at a fixed pixel, `I_t(p)-I_r(p)`, is **Eulerian**. It can contain a different material after motion. Use this only for points proven visible on matching material support and report invalidity separately; otherwise it is not `partial_t I` of the material surface. Camera poses are fixed in this dataset; if new moving cameras are supported in the future their known camera velocity terms must be included.

For inference prefer the **finite-displacement material-following residual** already implemented, but **change its parameterization/linearization** to expose velocity guidance instead of introducing duplicate observations. For reference 3D material point `X_r`, a current candidate `X_t` and current subpixel coordinate `u_t=pi_c(X_t)`:

\[
 r_I(X_t,X_r)=I_{c,t}(u_t)-I_{c,r}(u_r),\quad u_r=\pi_c(X_r),
\]

\[
 r_D(X_t)=D_{c,t}(u_t)-Z_c(X_t).
\]

The reference RGB may be an accepted **measured source pixel** `p_ref` offset from `pi(X_r)`. If keeping the existing material anchor convention (`u_t=p_ref+pi(X_t)-pi(X_r)`), use EXACTLY that convention and account for `X_r` in derivatives. Do not silently replace it with `u_t=pi(X_t)`; evaluate that as a separately labeled physical model. Reference geometry/depth uncertainty already encoded in the source state must not be double-counted.

For `u_t=pi(X_t)`:

\[
 H_{I,X_t}=\nabla I_{c,t}(u_t) J_{\pi,c}(X_t),\quad
 H_{D,X_t}=\nabla D_{c,t}(u_t) J_{\pi,c}(X_t)-J_{Z,c}(X_t).
\]

For the actual anchor convention `u_t=p_ref+pi(X_t)-pi(X_r)`, replace `H_{X_r}` appropriately with `-grad I_t J_pi(X_r)` and `-grad D_t J_pi(X_r)` (plus other explicit reference terms, if present); verify derivatives by finite differences of the *exact implementation*, including clean latent reference RGB-D.

**These finite-displacement Jacobians and equations (2.1–2.2) are two linearizations of the same appearance/depth physics**, not independent likelihoods. Using only the same pixel samples cannot create new information just by re-expressing them as velocity rows. Improvements can only come from more correct material support, temporal association, pooling independently observed image pixels, or justified shared part motion.

## 3. Velocity guidance for the existing free 6D `[x,v]` point filter

### 3.1 State and dynamics — unchanged

\[
 s_i=[X_i;v_i]\in\mathbb R^6,\qquad
 s_{i,t}^-=F(\Delta t)s_{i,t-1}^+,\quad P^- = FP^+F^T+Q(\Delta t),
\]

where `F=[[I,dt I],[0,I]]`, and `Q(dt)` exactly from `point_kalman_cuda.py` (including correct signed-backward behavior). Retain the prior and its covariance fixed over each IEKF inner solve. No artificial stationary attractor, velocity reset, gradient-induced motion prior or gate.

### 3.2 Do not incorrectly claim a new velocity Jacobian when the measurement is at current state

For residual `r(X_t)` and state `s_t=[X_t,v_t]`, the instantaneous residual Jacobian is **`[H_X, 0]`**. This is correct. The filter updates `v_t` through `P^-_{vX}` and temporal process factors. Making it `[H_X, dt H_X]` for that **same current-time state** is WRONG: it double-counts kinematic dependence already represented in the state.

If we parameterize the measured *next position* by a **previous-time** state (or an explicit velocity over the interval), `X_t≈X_{t-1}+dt v_{t-1}`, then

\[
 \frac{\partial r}{\partial v_{t-1}}=\Delta t H_X,
 \quad \frac{\partial r}{\partial X_{t-1}}=H_X.
\]

This is a coordinate reparameterization of the **same measurement**. Implement a separate velocity-information **diagnostic** from that chain rule and compare it with the original 6D Kalman update; do NOT use an extra independent pseudo-measurement with same raw pixels.

### 3.3 Actual per-camera fusion

For every point `i`, stack RGB 3 + depth 1 residuals from each admitted first-hit camera. Use the local-correlated within-point covariance machinery to include reference/current pixel re-use and state-reference covariance; do not sum independent likelihoods if they share raw pixels. Define the physically meaningful per-camera diagnostic:

\[
 G_{i,c} = \begin{bmatrix}\nabla I_{c,t}J_{\pi,c}\\ \nabla D_{c,t}J_{\pi,c}-J_{Z,c}\end{bmatrix}
 \in\mathbb R^{4\times3},
\]

and `G^v_{i,c}=dt G_{i,c}` when parameterized in prior velocity. For **independent rows**,

\[
 \Lambda_{v,i}=\sum_c (G^v_{i,c})^T R_{i,c}^{-1}G^v_{i,c},
 \quad b_{v,i}=\sum_c (G^v_{i,c})^T R_{i,c}^{-1}\,(-r_{i,c}).
\]

For general within-point correlated innovations use **one stacked** covariance `R_stack` (and cross-covariance if current states/reference noise correlate), `Lambda=G^T R_stack^{-1}G`. Report this as a **measurement information diagnostic**, distinct from posterior precision; signed velocity-space guidance must use the model's residual sign and proper prior mean. Compute eigenspectrum and 3D directions in world frame and in local tangent/normal basis. Compare each camera's `b` direction to the all-camera result and to post-hoc GT. Label rank deficient directions and uncertainty rather than inventing a measurement.

### 3.4 IMPORTANT: first-hit visibility and reference support

Projection of **all points** is used to compute the first-depth owner in each camera. An occluded point gets no direct row from the occluding pixel. A point may move across integer pixels; preserve its *material identity* and validate candidate ownership rather than enforcing fixed current integer pixel signatures. No background RGB/depth enters. Do not mistake a valid projected pixel for same-material temporal correspondence.

Track per point/camera: projected valid; foreground mask positive; first-depth winner; temporal source camera/frame/pixel; same-material support verified or unverified; depth discontinuity; temporal gradient admissible; accepted in final IEKF iteration. Publish a diagnostic matrix with every stage's drop count. **Lag=3 reference expiry** and no-row prediction must be explicitly audited: missing rows are not a zero-velocity measurement.

## 4. OPTIONAL shared moving-part motion — the *new physical information* experiment

This section implements a **separately opt-in**, part-wise motion model. It is the only route in this specification that can transfer motion information from textured/boundary points to textureless points. It is NOT automatically justified by image gradients and MUST NOT activate without validated part membership.

### 4.1 Rigid part instantaneous twist

For part `g`, choose a world-frame reference origin `o_g`; its instantaneous twist is

\[
 \xi_g=[v_{o,g};\omega_g]\in\mathbb R^6,
 \qquad v_i=v_{o,g}+\omega_g\times(X_i-o_g)
     =B_i\xi_g,
\]

\[
 B_i=[I_3\mid-[X_i-o_g]_\times]\in\mathbb R^{3\times6}.
\]

Pointwise RGB/depth temporal-gradient equations become

\[
 \boxed{\partial_t I_{i,c}=-\nabla I_{i,c}J_{\pi,c}B_i\xi_g},
 \quad
 \boxed{\partial_t D_{i,c}=(J_{Z,c}-\nabla D_{i,c}J_{\pi,c})B_i\xi_g}.
\]

Stack **all legitimate pixels and all cameras for all independently confirmed group members** into one group `H_g` and estimate **one** twist. For Blade's GT-declared prismatic motion, an oracle **translation-only** arm sets `omega=0` and estimates 3D translation, but **never copies the GT motion direction or amplitude into inference**. Add a general full 6D twist arm for USB (rotation). Both arms use **independently validated grouping**.

### 4.2 Nonlinear finite motion in SE(3): no first-order approximation for large steps

For a candidate displacement pose, use an explicit **pivot-centered world-frame chart** `X_i(t)=o_g+R_g(t←r)(X_i(r)-o_g)+t_g`. Let `C(o_g)` be translation by the chosen fixed reference origin. A left perturbation about `o_g` on the equivalent homogeneous world transform `T_g` is `T'_g=C(o_g) exp(hat(delta_zeta)) C(o_g)^(-1) T_g`; do **not** confuse this with an uncentered world-left perturbation, which would yield the rotational block `-[X_i]_×` rather than `-[X_i-o_g]_×`:

\[
 T'_g=C(o_g)\exp(\widehat{\delta\zeta})C(o_g)^{-1}T_g,
 \quad \frac{\partial X_i}{\partial\delta\zeta}=
 [I_3\mid-[X_i-o_g]_\times]=B_i(X_i),
\]

for the corresponding centered left-action chart. If the optimization variable is **twist velocity** `xi` with `T=exp(dt * hat(xi))`, include the exact **`SE(3)` exponential left/right Jacobian** and `dt` by chain rule: `dX/dxi ≠ dt B` for finite rotation in general. Specify **which** frame the twist is expressed in, whether reference origin moves, and whether `xi` is velocity or pose increment. Independently finite-difference check both translation and rotation at nonzero angles; never mix left and right Jacobians.

Per actual current image sample `u_t=pi_c(X_t)`:

\[
 H^I_{g,c,i}=\nabla I_{c,t}(u_t)J_{\pi,c}(X_t)\,\frac{\partial X_t}{\partial\zeta_g},
\]

\[
 H^D_{g,c,i}=(\nabla D_{c,t}(u_t)J_{\pi,c}(X_t)-J_{Z,c})\frac{\partial X_t}{\partial\zeta_g}.
\]

Use actual material-following source pixel convention if different from `pi(X_t)`; derive its reference-state derivatives rather than dropping them. Verify exact residual and physical 3D motion on oracle point identities; image alignment is **not** independently valid merely because a set of points is labeled the same part.

### 4.3 Three experimental membership arms (NO GT leakage)

- **Arm P — Free point:** per-point independent `[x,v]`; identical visible observations; new velocity/tangent-normal diagnostics, no shared state. **Control**.
- **Arm G — Offline oracle group:** **GT part IDs only in a separate audited offline executable**, never as filter input; estimate rigid/prismatic twist from measured RGB-D on 0→1 and 1→2. This answers: *if physical grouping were correct, do existing images provide enough multiview part-motion information?* Report oracle privileges and prevent any GT values from influencing parameter selection for non-oracle runs.
- **Arm C — Actual non-GT group:** **Only if** the existing `coherence` stage (or another independently observed non-GT membership file) supplies explicit verified member IDs and provenance. Use its actual grouping unchanged initially; **do not derive perfect groups from GT** or silently assign ungrouped static points to moving parts. If downstream coherence requires accurate F1 tracks and provides no usable groups, mark `PART_MEMBERSHIP_UNAVAILABLE`, skip Arm C, and leave the new part-wise mode not production-ready. A plain fixed pixel neighborhood is not sufficient evidence of a rigid part.

Shared twist implies **cross-point posterior correlation**. A correct group posterior is `P_xi` and per-point covariance `J_i P_xi J_i^T + P_cond,i` if extra independent point uncertainty is explicitly modelled. Retain the shared latent twist as a joint factor; **do not** repeatedly insert its guidance as independent point observations and then claim independent correct 6×6 covariance. Group covariance must be checked against independent dense or QR oracles on tiny scenes. Preserve the ability to use point-local updates for all non-group points.

### 4.4 Protect static points physically, not with a hidden zero-speed gate

Zero-twist is a legitimate hypothesis `xi=0`: for a stationary surface with constant illumination and no occlusion, predicted `partial_t RGB=0` and `partial_t D=0` despite nonzero spatial gradients. The static-control tests must recover near-zero velocity under physically valid images and express honest uncertainty where image information is absent. **Do not classify points static by GT in inference and freeze them.**

For any *part* grouping, explicitly test erroneous membership of static and moving points. A single incorrect merge can drag static points. Such cases must produce a documented `GROUP_IDENTITY_INVALID` diagnostic, not be considered successful tracking. If group identity is unknown, keep independent point filtering.

## 5. Statistical correctness and gradient uncertainty

**First derive the likelihood of the actual image observations**, including raw RGB-D noise and reference-state/pixel uncertainty. The measured current RGB/depth uncertainty in Blade is extremely small relative to some interpolation/occlusion discrepancies. Never pretend `R_warp=0` is physically calibrated. Preserve raw pixel IDs and original depth uncertainty; no source-depth or reference noise counted twice.

If using `partial_t I ≈ (I_t(p)-I_r(p))/dt` at fixed same-material pixel, its noise is **temporally correlated** (references reused) and typically scales like `1/dt^2` in variance. Spatial image gradients computed from common raw pixels also share sensor noise with temporal derivatives. The nominal row `G` therefore has **errors in the regressor**, not simply additive independent RHS noise. The separate differential solver is an *approximate diagnostic* unless an error-in-variables model is derived and validated. For actual inference, prefer finite-displacement residuals evaluated using actual source/current raw pixel latents and their joint covariance.

Sensor model, bilinear quantization, geometric interpolation bias, identity mismatch and temporal appearance variation must be separated in the report. Do not simply add a large scalar `sigma_warp` to improve NEES, artificially lower information, or zero static drift. A covariance is only credible if calibrated by independent controls/held-out measurements. Evaluate RGB-only, depth-only and RGB+D ablations; account for reuse and RGB-depth correlations where measured.

Derive residual Jacobian wrt image sampling coordinates analytically and confirm with independent centered differences **away from interpolation cell boundaries**. At cell crossings, report piecewise differentiability and update sample support consistently. Do not retain invalid or newly occluded pixel residuals merely to maintain derivative continuity; equally, do not declare support-loss as objective improvement.

## 6. Concrete new code and integration

Suggested structure, renaming only if local code already provides the same role:

| Suggested file | Actual implementation |
|---|---|
| `estimator/temporal_image_gradients.py` | GPU-batched mask/depth-valid 2D RGB (3×2) and depth (1×2) derivatives at subpixel material positions; no cross-discontinuity gradient; return source raw-pixel provenance, image support and uncertainties. Reuse existing frame/sampler whenever possible. |
| `estimator/temporal_motion_geometry.py` | World `J_pi`, `J_Z`, depth-derived local `n,T,B` (diagnostic only), `G_I,G_D`, normal-vs-tangent component check and per-camera information matrices. No new persistent point normal state. |
| `estimator/temporal_motion_observability.py` | Stack camera rows with their correct within-point covariance; per-camera gradient/rhs/information, joint eigen/singular directions, camera conflict audit, missing-row breakdown. |
| `estimator/temporal_gradient_iekf.py` | Opt-in actual **replacement likelihood** using finite-displacement material-following residual and correct `H`/reference-state derivatives. Hook into uncommitted local-correlated filter; no duplicate gradient pseudo-measurements. Preserve old exact material-following path as control. |
| `estimator/temporal_part_twist.py` | Opt-in shared `SE3` twist factor over *known, explicit member IDs*; GPU-batched part-level small solves, covariance of group latent, and exact pose derivatives. No untrusted auto-membership. |
| `tools/audit_rgbd_gradient_motion.py` | GT only for offline checking; produce camera-by-camera tangent/normal velocity information and point traces, reference/visibility source IDs, residual-vs-velocity curves. May read run outputs but never mutate them. |
| `tools/audit_rgbd_part_motion_oracle.py` | Offline GT-membership `oracle` and actual explicit non-GT control arms; no leakage into deployable inference. |
| `tests/test_temporal_image_gradient_geometry.py` | Deterministic independent derivatives and geometry oracle. |
| `tests/test_temporal_image_gradient_observability.py` | Rank, covariance, camera combination and static/dynamic tests. |
| `tests/test_temporal_gradient_iekf.py` | Accepted frame covariance, reference pixels, first depth, no double count. |
| `tests/test_temporal_part_twist.py` | Prismatic/revolute SE3, mixed groups, uncertainty, small dense oracle. |
| `reports/temporal_gradient_motion_validation.md` and `.json` | Real metrics, commands, source hashes, math/code/test map, status and blockers. |

CLI proposal (implement if absent, verify exact final parser):

- `--temporal-gradient-guide off|diagnostic|replace` (default `off`): `off` bitwise-compatible old material-following; `diagnostic` computes but never updates state; `replace` uses new physically equivalent, verified finite-displacement residual/Jacobian in the point-local filter. Do not call it novel independent motion information if it uses same RGB-D pixels.
- `--temporal-part-motion off|non-gt` (default `off`): optional non-GT group membership only with explicit provenance and coherent stage input. GT oracle must be a separate offline tool (no `gt` option in `train.py`).
- `--temporal-part-membership PATH` required for `non-gt`. Validate file metadata and IDs against current canonical point IDs, frame order, camera/mask source hashes. No invented groups.
- `--temporal-motion-report-point-ids` optional original IDs for detailed traces; default selected known original IDs `blade_103706:14104,27016` and `USB_100109:67827,85696,87290` only when present.

Wire flags through `tools/run_temporal_rgbd_critical.py`, `train.py`, and manifest source hashes; **check current uncommitted CLI** because public HEAD does not advertise these modes. Continue to render only frontmost measured points and save all canonical IDs exactly; frame journaling/NaN tails unchanged.

## 7. Required tests and independent numerical witnesses

**Gate A — continuous geometry and mathematical exactness (CPU float64 then CUDA):**

1. Independent NumPy camera projection `J_pi` and `J_Z` against analytic and finite differences. Include off-axis cameras from Blade calibration, multiple depths, nonzero scene translation, reverse sign/dt, row/column pixel convention, actual 59 Hz timestamps.
2. Synthetic textured moving planar surface and static textured plane: check `I_t + grad I J_pi v≈0` at material-valid interior pixels, and `D_t − (J_Z-grad D J_pi)v≈0`, with independent rendered analytic data and known ground truth. Include an **inclined** plane (world normal vs camera Z not equal).
3. Depth row `a=J_Z-grad D J_pi` is proportional to geometric true surface normal for smooth plane, and `a@T≈0`; numeric derivatives match world-frame equations. At depth discontinuity/invalid 2×2 mask support, no derivative row is admitted. No normals stored in filter state.
4. Tangent-plane RGB: textured gradient in one direction has rank 1, orthogonal gradients from independent valid camera/pixels give rank 2, constant RGB creates zero tangent information; positive temporal difference alone must not generate false motion without a valid spatial flow/identity relation.
5. Exact `SE3` left pose-increment Jacobian and exact velocity-to-pose exponential chain rule vs independently implemented 3D rigid transform finite differences for translations and substantial rotation angles. For pure prismatic `omega=0`, all grouped points get equal world translation; for revolute, `v_i=v_o+omega×(X_i-o)`.
6. Static textured surface with strong spatial gradient and **no temporal change** yields zero data-driven velocity RHS (under correctly specified noise), both one and eight cameras. Add varying illumination to falsify perfect brightness constancy; report `PHOTOMETRIC_MODEL_INVALID` rather than moving the static geometry.

**Gate B — visibility, identity, statistical correctness:**

7. Two surfaces overlap in image: frontmost moving vs rear static (and reverse); only frontmost valid material generates image residual; background depth/RGB must never update a foreground point. Same-surface, silhouette, disocclusion, reference reappearance and positive masks; report all drop reasons.
8. A moving point crossing one or more integer pixel boundaries remains eligible when truly frontmost; the new path must **not inherit fixed-current-pixel acceptance**. Rerender ownership on candidate state, monitor competitors and do not benefit from silently dropping failed rows.
9. Same reference pixel used by one point across multiple frames is one local stochastic variable. Duplicate camera/pixel/temporal-difference noise cannot be assimilated twice. CPU exact local QR and CUDA parity of within-point posterior mean/covariance, including reference `Cov(X_t,X_r)` and `Cov(X_t,noise_key)`.
10. Compare original material-following vs `diagnostic` mode: **exact bitwise identical** trajectory, covariance, accepted rows, status and journal (apart from the new diagnostics) in a miniature deterministic scene. `replace` finite displacement must pass residual/Jacobian parity with the same physical model on fixed support before its behavior can be evaluated as a distinct implementation.
11. Test group posterior: shared twist induces cross-point covariance, and posterior point marginal `J_i P_xi J_i^T` matches independent dense Gaussian oracle. Falsify naive independent per-point replay of the same group measurement (show artificially small variance). Actual group identities must be complete/no duplicate IDs and non-GT-sourced.
12. Static+moving **incorrect group** produces visible false static motion unless prevented by correct membership: fail with `GROUP_IDENTITY_INVALID`, or report actual observed failure—do not hide behind GT-based in-inference class labels, hard velocity zero, or a speed threshold.
13. Test missing camera references after lag expiry, repeated valid camera handovers, signed backward timestamps, exactly no-row process-only propagation, and correct covariance inflation after no observation. Preserve all original point IDs between forward and backward clouds even though their endpoint catalogues differ.

**Gate C — physical scientific experiments (GT only offline):**

14. Independent **single-point gradient diagnostic** on Blade, original IDs 27016 and 14104, frames `0→1, 1→2, 4→7, 23→26`, all eight cameras. Save raw RGB-D gradients `I_u,I_v,I_t,D_u,D_v,D_t`, projection `J_pi`, depth `J_Z`, derived `n/T`, valid/invalid reasons, per-camera information eigenspectrum, proposed 3D velocity and final accepted state. Identify the exact row(s) that create >80 mm correction or 6 m/s estimated velocity if reproduced; do not assume a historical trace remains identical after code changes.
15. Independent **oracle-part** 0→1 and 1→2 for Blade prismatic and USB revolute. For the same material points, compare (a) individual point velocities, (b) oracle part 3D translation, and (c) oracle full 6D twist where physically appropriate. GT defines **membership only** for this arm; GT transform/amplitude/axis must be excluded from the numerical estimator. Estimate from RGB-D and compare to GT after estimation. Include a zero-velocity true-static group.
16. Camera ablation: same points and same valid raw observations, best one camera, complementary 2-camera pairs, 4, 8. Report counts **and** `sigma_min`, condition number/observable subspace, measured vs oracle velocity angular error, normal/tangent component error, and per-camera signed gradient/RHS contributions. If only 1–2 cameras are admitted, prove whether all eight geometrically valid views were truly available or lacked references/material support.
17. RGB-only, depth-only, RGB+depth ablations: separately quantify surface-tangent and normal velocity errors; include real textureless region and textured region. No claim that a flat untextured surface has observable tangent velocity from RGB.
18. Objective test at `v=0`, `v_GT` and independent random physically plausible directions (GT solely offline) using **identical pixel support**: does the observation likelihood actually prefer correct motion? Also evaluate full production candidate support changes separately, so row disappearance is not an artificial improvement.
19. Static drift: p50/p90/p99 translation and estimated speed over frames 0,1,2,3,7,30,59; no stale-covariance “success.” Dynamic: median/p90/p99 3D error, velocity error, accumulated displacement drift, NEES and uncertainty *with the correct cross-point approximation qualification*.
20. Performance: compare `off`, `diagnostic`, `replace` on 1k points and 3 real frames, then full-resolution USB/Blade (8 cameras) if earlier gates pass. Measure rows per point, valid camera coverage, runtime/frame, GPU memory, covariance SPD, number of accepted/rejected updates, and point IDs preserved. No globally correlated full Schur triggered by opt-in gradient guidance.
21. Regression: all existing 129 material-following focused tests *if present locally*, plus the full existing `tools/test_temporal_rgbd.py` (may vary on current branch) and new tests. Do not silently delete old failing tests or lower required precision.

**Tests must use independent reference models and known exact analytic images** (e.g. affine RGB textured plane and analytic perspective depth) rather than only comparing two functions from the same implementation. For actual camera warps use independent NumPy finite differences and single-point CPU QR/SVD oracle. Report experimental result `MOTION_UNIDENTIFIABLE` rather than fabricating velocity when rank deficient.

## 8. Scientific gate order — stop on invalid physics, not on elapsed optimism

1. **Gate 0** — inspect real uncommitted code, map equations to existing functions, record baseline source and result hash; run old tests without code changes.
2. **Gate A** — implement and independently verify gradient extraction, world projection, tangent/normal interpretation and exact rigid `SE3` derivatives.
3. **Gate B** — implement **diagnostic-only** per-point velocity information; test mask/first-hit/same-material and no-double-count. Run on old saved Blade full 60-frame output **read-only** to identify camera guidance and dropouts, not claim tracking success.
4. **Gate C** — implement and independently test opt-in **replacement** gradient-linearized finite material-following likelihood; run **3 frames** Blade and USB, compare static/dynamic errors against exact old material-following, using post-hoc GT scoring. Explicitly state if the new path is merely equivalent and adds no information.
5. **Gate D** — separate GT-only oracle part membership analysis of Blade prismatic and USB revolute; use measured RGB-D exclusively to **solve** motion. If oracle grouped points recover true motion and maintain static control, provide evidence that **correct shared part identity** is the missing structural information. If oracle group fails, stop; investigate material identity/likelihood before non-GT grouping.
6. **Gate E** — optional `non-gt` part motion only if actual existing non-GT grouping is available, correct and reproducible. Independently validate results. **Never create automatic GT memberships for production.**
7. **Gate F** — up to 11 frames only if 3-frame moving-point error improves, static p99 drift does not materially deteriorate, posterior covariance is consistent, and GPU memory supports the workload. No 60-frame 10-scene batch yet.

**No arbitrary success thresholds**: report measured error and confidence intervals alongside unchanged baseline, predeclare comparison metrics, and determine whether change is statistically meaningful on same point selection. A direction that is not observed must not produce unjustified confident movement.

## 9. Proposed reproducible commands (only after implementing/confirming CLI)

First run tests (CPU float64 independent oracle; CUDA parity where available):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B -m unittest -v \
  tests.test_temporal_image_gradient_geometry \
  tests.test_temporal_image_gradient_observability \
  tests.test_temporal_gradient_iekf \
  tests.test_temporal_part_twist
python -B tools/test_temporal_rgbd.py
```

Standalone offline audit of recent Blade output (must not modify the original run):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
python -B tools/audit_rgbd_gradient_motion.py \
  --scene blade_103706 \
  --run-root '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_20261102_140950_2d1ffaf3/blade_103706_material-following' \
  --point-ids 14104,27016 \
  --frames 0,1,2,3,4,5,6,7,23,24,25,26,30,59 \
  --output-root '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/gradient_motion_audit_blade'
```

Opt-in **diagnostic** pilot, existing material likelihood unchanged, first 3 frames, all cameras/points:

```bash
python -B tools/run_temporal_rgbd_critical.py \
  --mode material-following --direction forward --frame-limit 3 \
  --scenes blade_103706,USB_100109 --stop-after free \
  --pcd-subsample-factor 1 --temporal-lag-frames 3 \
  --sensor-noise measured --temporal-warp-noise ideal \
  --temporal-gradient-guide diagnostic
```

Opt-in **replacement** pilot only after diagnostic/oracle tests:

```bash
python -B tools/run_temporal_rgbd_critical.py \
  --mode material-following --direction forward --frame-limit 3 \
  --scenes blade_103706,USB_100109 --stop-after free \
  --pcd-subsample-factor 1 --temporal-lag-frames 3 \
  --sensor-noise measured --temporal-warp-noise ideal \
  --temporal-gradient-guide replace
```

The above new gradient flags, audit tool and tests **do not exist in the verified public commit**; they are required implementation targets. The `ideal` noise flag is for controlled comparison only, **not** calibrated physical validation. The offline GT-part oracle script must print its own verified usage after implementation and strictly disallow GT part labels inside `train.py`. If local CLI has changed, Codex must **print the exact verified command** rather than silently dropping/renaming options.

## 10. Mandatory report and handoff

Create `reports/temporal_gradient_motion_validation.md` and `.json`, with:

- HEAD, uncommitted state, affected file hashes, exact CLI, input/output paths, test counts, and independently reproduced baseline results.
- Equation-to-code-to-test table for (i) camera projection, (ii) RGB differential, (iii) depth differential and normal, (iv) point velocity chain rule, (v) part `SE3` twist, (vi) covariance/visibility/reference policy.
- All eight per-camera RGB/depth contributions at selected points and frames; show `G_I`,`G_D` or standardized summary, image gradient support, temporal RHS, measured residual, covariance and 3D information eigendirections.
- Separate **geometric visibility** / **first-depth winner** / **foreground support** / **valid material correspondence** / **valid temporal reference** / **accepted Jacobian rows**. Histograms by frame, especially after 3-frame lag expiry.
- Static vs moving position/velocity errors on **identical original IDs** and their posterior uncertainty (including NEES); camera ablation and RGB/depth/combined ablation.
- Oracle-part results with GT used only for membership and post-hoc evaluation; `non-gt` part results only if available. Correctly label any missing memberships/measurements.
- Covariance equivalence/non-equivalence to material-following baseline; no image-gradient duplicate likelihood or false claim that reparameterization adds sensor information.
- Full test outputs, runtime/GPU peak, viewer panels, exact source hashes and remaining physical/statistical blockers.

**Final evidence-based questions to answer:**

1. Is the same material's RGB/depth temporal change consistent with the actual physical 3D velocity at GT? Under what camera/surface conditions does it fail?
2. Does projecting image-plane RGB gradients into the 3D surface tangent plus depth-normal velocity system produce a *correctly directed* multiview 3D velocity observation for the independently tracked point? How frequently is tangent velocity rank-deficient?
3. Can correct shared moving-part membership pool complementary gradient information across material points to recover prismatic Blade motion and revolving USB motion **without moving static points**?
4. If static drift appears, which exact camera/pixel/material-identity or covariance assumption created the first false velocity impulse? If moving parts still fail, label the specific missing information and do not claim convergence is accuracy.

**Success means improving real moving-point velocity/position estimates and preserving static stability, not passing mathematical unit tests alone.**
