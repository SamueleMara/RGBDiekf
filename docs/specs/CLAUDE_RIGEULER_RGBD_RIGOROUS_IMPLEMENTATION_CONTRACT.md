# CLAUDE CODE — AUTHORITATIVE IMPLEMENTATION CONTRACT
## RigEuler3DGS methodology as a mathematically consistent rendered-image RGB-D IEKF for colored XYZ points

**Date:** 2026-10-24  
**Target working tree:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Reference:** `SamueleMara/RigEuler3DGS`, pinned commit `481b2cf2d5602932286948cf14114f9caee4de3c`  
**User requirement:** implement the theory rigorously. **Do not preserve a method simply because a previous blade experiment happened to return the correct joint.** Inspect every relevant equation, Jacobian, covariance, noise assumption and decision rule; replace incorrect components surgically or completely as necessary. Keep the prior run as a comparison, not as authoritative mathematics.

### 0. Nonnegotiable interpretation

The estimator is an **iterated extended Kalman filter (IEKF)**. The state predicts a transported colored point cloud. A differentiable RGB-D point renderer maps this state to **predicted RGB/depth images in all calibrated cameras**. The **actual next-frame RGB/depth images** are observations. The IEKF uses their **innovation**, its measurement Jacobian, the **predicted covariance**, an iterated correction, and a **posterior covariance** carried to the next frame.

There is **no nearest-neighbor point-to-point association as the observation equation**, no optimized background color, no 3DGS alpha compositing, no Gaussian opacity, and no photometric training of colors. An image-space MAP/quadratic objective is permitted **only** as the algebraically equivalent local solve / nonlinear trust-region acceptance of the **same IEKF**, as in the reference source. It must not replace recursive prediction or covariance propagation.

Implement **all cameras at every frame** for forward and backward filtering, rigid fitting, and joint refinement. The cross-validation used in RigEuler's motion coherence separates **representative nodes**, not camera subsets.

Keep three kinds of state distinctly named:

1. **Physical free state:** independent point positions and velocities (6N). Each point is independently propagated; computational groups are *not* rigid objects.
2. **Computational tangent basis:** group-wise 12D corrections, expanded to pointwise 6D increments. The groups are held fixed in canonical geometry; they do not constrain the physical trajectory across frames.
3. **Single physical moving-body state in Phase 2:** a shared rigid SE(3) pose and spatial twist (12D); only the coherence-selected points are transported with this shared pose.

Do not invent extra physical parts, persistent group rotations, unknown K-moving-body discovery, masks of arbitrary moving fraction, graph splits from motion magnitude, or a different estimator. All transforms, depths, velocities and camera coordinates must use recorded real units (meters, seconds, camera intrinsics/extrinsics).

### 1. Read and map the actual source before editing

Read, in this order, the **exact files at the pinned commit** (not a different HEAD):

| Reference file | Mandatory mechanism |
|---|---|
| `estimator/process.py` | State prediction and error-state transition; reduce 12D Gaussian state to 6D XYZ+v |
| `estimator/covariance.py` | Prediction, invertibility, PSD checks, **approximate blockwise posterior**; document which parts are *not* exact |
| `estimator/renderer_bridge.py` | One synchronized multiview `FrameLinearization`; `predicted`, `apply_H`, `apply_HT`, block Fisher |
| `estimator/sparse_iekf.py` | `iekf_frame_step`, `measurement_operators`, fixed predicted prior, grouped tangent solve, trust region, final posterior |
| `estimator/supernodes.py` | Geometric computational grouping; matrix-free `G`, `Gᵀ`, not physical part assignment |
| `estimator/motion_coherence_support.py` | Time-persistent log-distance strain and fixed canonical kNN graph |
| `estimator/coherence_extraction.py` | Candidate hierarchy, complement-relative motion, node-fold cross-validation, selection and optional recruitment |
| `tools/run_motion_coherence_support.py` | **Independent fixed-count coherence representatives:** default 192, kNN=12; not Phase-1 occupancy |
| `estimator/rigid_part_runtime.py` | `RigidState` and `run_rigid_window`, Kabsch initialization, pose/twist process and covariance |
| `estimator/joint_iekf.py` | 12D small-state fixed-prior iterated correction, exact SE(3) prior error and Jacobian, final information posterior |
| `joint_reduction/rigid_trajectory_reduction.py` | Relative spatial increments, `gamma_1dof`, finite-axis P/R fitting, pose-space refinement; **no image renderer in reduction** |
| `tools/run_rigid_se3_pipeline.py` | Stage order; freeze the moving core before rigid filtering, reduce joint only after rigid trajectory |

Map these against the **actual local Claude working files**, including:
`pointwise_rgbd/rgbd_point_renderer.py`, `image_observation.py`, `image_iekf.py`, `image_phase1.py`, `image_coherence.py`, `rigid_image_iekf.py`, `joint_image.py`, `image_bidirectional.py`, `image_support.py`, `rigeuler_ref/*`, `tools/run_rgbd_image_iekf.py` and `tests/test_image_*`.

**Deliver before implementing:** `reports/theory_to_code_audit.md`, listing each reference equation, reference source function, local implementation/function, mathematical equivalence or deviation, expected dimensionality, and its finite-difference/dense-reference test. **Do not declare something equivalent merely because it outputs similar results.** The earlier Claude Code work is baseline evidence only; it is not exempt from reimplementation.

### 2. Exact mathematical model — free point state

For point `i` at time `t_k`:

```
s_i,k = [x_i,k (3), v_i,k (3)]ᵀ ∈ R⁶
s_k = stack(s_1,k, ..., s_N,k) ∈ R^(6N)
color_i ∈ [0,1]³ : fixed canonical appearance
rho_i > 0 : fixed geometric screen footprint derived from input point sampling
```

All point colors are samples from the actual endpoint RGB-D images (consistent RGB convention, frozen once created). **Color is not a free variable in the IEKF**; do not move the RGB measurement into a correspondence heuristic.

**Predict exactly once** per frame with actual `dt = t_k - t_(k-1)`:

```
x_i,k^- = x_i,k-1^+ + dt * v_i,k-1^+
v_i,k^- = v_i,k-1^+
F_i = [[I3, dt*I3], [0, I3]]
P_k^- = F P_(k-1)^+ Fᵀ + Q_k
```

If using integrated white acceleration with spectral density `q_a` (3×3 diagonal/isotropic), discretize **consistently**:

```
Q_i = [[(dt³/3) q_a, (dt²/2) q_a],
       [(dt²/2) q_a, dt q_a]]
```

Optional direct position random-walk noise requires a separately documented continuous-time density; do not smuggle a per-frame constant into `Q`. Reverse processing uses its own correctly time-reversed prediction convention and positive elapsed duration; reverse the frame order **without silently inverting the meanings of canonical geometry, velocity or world-frame timestamps**. Test the signs and transition Jacobians by finite differences and forward/backward analytic examples.

The state is **not** a 12D SE(3) state per point: there is no per-point orientation or angular velocity. Only the computational correction basis has virtual rotation variables.

### 3. Measurement function is the rendered next-frame RGB-D, not associations

At iteration `j` of frame `k`, use the current transported point state to compute for **every camera `c`**:

```
L_c(s^j) = render_color_depth_coverage({x_i^j, color_i, rho_i}, camera_c)
         = (Ihat_c^j [3,H,W], Dhat_c^j [H,W], Ahat_c^j [H,W])
```

The actual measurement `y_c,k = (I_obs,c,k, D_obs,c,k, valid_rgb, valid_depth, foreground_mask)` is immutable for that video frame. Define the **innovation** on valid observed and predicted rays:

```
r_I,c^j(p) = I_obs,c,k(p) - Ihat_c^j(p)
r_D,c^j(p) = D_obs,c,k(p) - Dhat_c^j(p)
```

Coverage `Ahat` is **predicted support/visibility**, not rendered background and not Gaussian opacity. `Ahat` may control whether a predicted depth/color exists, and its derivatives may be used only if an explicit sensor/foreground-coverage observation model is justified. For pixels with no prediction, do **not** synthesize an RGB/depth background measurement. Expose the observed-foreground versus rendered-coverage mismatch separately, without silently counting missing pixels as a zero innovation. Handle occlusion, overlap and footprint boundaries from the actual point-cloud rasterizer.

Implement a mathematically specified point rasterizer: camera projection, positive-depth clipping, nonnegative differentiable footprint weights, deterministic front surface and visible contributor set, normalized color and metric-depth aggregation, image coverage. If the chosen front-selection changes discontinuously, its Jacobian is **piecewise**; freeze the selected contributors **within a single linearization**, rerender and recompute contributors for every *accepted trial* and next iteration, and test errors at ordering boundaries explicitly. No invented 3DGS alpha or learned opacity.

**Mandatory renderer API (identical conceptual role as RigEuler):**

```python
class RGBDFrameLinearization:
    def predicted(self): ...                 # all-camera (rgb, depth, coverage)
    def apply_H(self, point_xyz_delta): ... # JVP, pixel RGB-D changes
    def apply_HT(self, pixel_adjoints): ...  # VJP, sums ALL cameras to xyz
    def build_block_fisher(self, weights): ... # per-point 3x3 diagonal blocks, PRECONDITIONER ONLY
```

The observation Jacobian is

```
H_j = ∂h/∂s |_(s^j) = [∂h/∂x, 0_(pixels×3N)]
```

because a frame's image directly observes **positions**, not instantaneous velocities. Velocity is corrected through the propagated position–velocity cross-covariance. The rasterizer JVP and VJP must be exact adjoints of the same chosen forward model at the same frozen contributor structure. In particular:

```
< H u, z > = < u, Hᵀ z >
```

Use multicamera additive information, and include pixel-to-multiple-point coupling in `Hᵀ W H` (never replace the normal operator with a sum of independent single-point blocks for the correction).

**Noise assumptions:** `R_k` is the observation noise covariance of the actual RGB-D images; report separately RGB and depth noise, depth-range dependence, mask validity, model-mismatch covariance and any pixel correlations. Weighted innovations can use the reference's specified robust IRLS weights if supported by a matching likelihood. Do not count neighboring footprint pixels as independent evidence for a calibrated p-value unless their residual dependence is accounted for. No arbitrary noise adjustment to force a joint decision. Calibration/intrinsics/extrinsics and timestamps are fixed inputs unless a *separate* explicit calibration model exists.

### 4. The RigEuler Phase-1 IEKF: predicted prior fixed, image innovation corrected iteratively

A plain `sum(r_image²)` is **not an IEKF**. Implement the actual recursive IEKF. For compact exposition, let

```
s_minus = predicted state,  P_minus = predicted covariance
Lambda_minus = P_minus^{-1}
s_j = current nonlinear IEKF iterate
r_j = y_k - h(s_j)
H_j = ∂h/∂s |s_j
W_j = R_j^{-1}
e_j = s_j - s_minus                  # Euclidean free-point error
```

At a fixed linearization, the **unrestricted** information form of the iterated Kalman step is

```
A_j(delta) = Lambda_minus delta + H_jᵀ W_j H_j delta
b_j = H_jᵀ W_j r_j - Lambda_minus e_j
solve A_j delta_j = b_j
s_(j+1) = s_j + delta_j                # after acceptance
```

This is precisely the reference `measurement_operators`/`iekf_frame_step` normal operator and corrected-prior RHS, specialized to XYZ and velocity. The equivalent ordinary IEKF recursion is

```
K_j = P_minus H_jᵀ (H_j P_minus H_jᵀ + R_j)^-1
s_(j+1) = s_minus + K_j [ y - h(s_j) + H_j(s_j - s_minus) ]
```

The *implementation* may use matrix-free PCG instead of explicitly forming `K_j`: the equations are equivalent when the same prior and observation model are used. **Do not call a standalone image optimizer an IEKF.** Do not update `P_minus` or count the same observation again inside successive iterations.

#### Grouped correction exactly as a tangent-space restriction

Fix a canonical computational partition. For group `g` with center `c_g`, define a **12D group correction** 
`zeta_g = [dt_g(3), dphi_g(3), du_g(3), domega_g(3)]`. For each point `i` of that group:

```
dx_i = dt_g + dphi_g × (x_i^canonical - c_g)
dv_i = du_g + domega_g × (x_i^canonical - c_g)
```

Use the exact local code convention for cross products (`dphi × r = -[r]× dphi`), with a finite-difference test on **G and Gᵀ**. If you choose updated rather than canonical lever arms, prove Jacobian consistency and chart resetting. **These are correction coordinates only**; do not keep an SE(3) object per group between frames.

Write `delta = G zeta`. The required restricted **iterated** Kalman equations are

```
A_G,j(zeta) = Gᵀ [ Lambda_minus (G zeta) + H_jᵀ W_j H_j (G zeta) ]
b_G,j = Gᵀ [ H_jᵀ W_j r_j - Lambda_minus e_j ]
A_G,j zeta_j = b_G,j
s_trial = s_j + G zeta_j
```

The reference's trust region / Steihaug PCG may add an explicitly **step-regularization** term for convergence, but it must not be silently stored as measurement information in the Bayesian covariance. Build block Fisher **only as preconditioner/observability diagnostic**, while the **operator itself** evaluates pixel couplings through `apply_H`/`apply_HT`.

At each iteration:

1. Start from the once-predicted `s_minus`, `P_minus`.
2. Render all cameras; get innovation and exact `H_j`, `H_jᵀ` operators.
3. Form `b_G,j`, matrix-free `A_G,j` and solve the grouped correction using the RigEuler trust region / CG.
4. Apply the trial correction; **rerender the actual nonlinear predicted RGB-D images**.
5. Compute the same fixed-prior + measured-RGB-D probability terms for trial acceptance (as in RigEuler `evaluate_true_map_energy()`); reject/accept based on the *same model*, no changing goals or hidden per-frame priors.
6. If accepted, relinearize at the new state. Stop using the reference gradient/step/innovation convergence criteria, recording the actual termination reason.
7. After the final accepted iterate, update the posterior **ONCE**, and propagate it into the next frame.

**A nonlinear trust-region scalar used for step acceptance is not an extra training loss.** It is the evaluation of the same Kalman observation likelihood and fixed predicted prior; explicitly document that distinction.

### 5. Covariance: do not misrepresent RigEuler's approximation or the grouped posterior

**RigEuler reference behavior:** its Phase-1 correction solve is matrix-free and captures cross-Gaussian image coupling, but its Phase-1 posterior is **approximate blockwise information inversion** (`approximate_posterior_information` with per-Gaussian Fisher), NOT the exact global posterior. RigEuler's Phase-2 compact rigid IEKF inverts the **full small 12×12** final information matrix with the correct nonlinear-prior Jacobian.

**Our mathematical requirement:** because group-constrained correction steps operate in a smaller tangent subspace but the full physical point state remains 6N, simply setting

```
P_plus = (Gᵀ[Lambda_minus + Hᵀ W H]G)^-1
```

and calling it the point posterior is **wrong**: that is a reduced solve-space matrix, not the covariance of the 6N physical states. Likewise, the naive per-point block inverse is an approximation that can understate variance (the prior report measured ~20% on a small dense case).

At the **final, accepted linearization**, the restricted linearized state gain is

```
B = Gᵀ(Lambda_minus + Hᵀ W H)G
K_G = G B^-1 Gᵀ Hᵀ W
```

The linearized covariance of **this chosen restricted-gain estimator**, given full input covariance `P_minus` and `R`, is

```
P_plus = (I - K_G H) P_minus (I - K_G H)ᵀ + K_G R K_Gᵀ
```

This is a **Joseph covariance for the implemented restricted estimator**. It is not necessarily the posterior of an unrestricted full-state Bayesian filter. Distinguish covariance of the chosen estimator from true Bayesian uncertainty under a different prior/constraint model.

**Required engineering decision and proof:** For small fixtures, implement exact dense references for `K_G`, `P_plus`, prediction and accepted-point covariance. For the complete RGB-D scene, provide a matrix-free/compressed representation only if its action, PSD property, and complete multiframe recursion can be proven for the actual **cross-group couplings** created by shared rendered pixels. Do not assume a fixed rank-24-per-group covariance form remains closed under a global image measurement coupling unless verified algebraically and numerically. Do not silently drop off-block terms, reset to an initial P each frame, or call any block-diagonal marginal propagation exact. If a bounded approximation is required to run on 8 GB, isolate it behind a named interface, quantify its error on increasingly large dense-oracle fixtures, propagate the approximation honestly, and publish its consequences for inferred confidence.

**Acceptance checks:** symmetry, PSD/eigenvalue diagnostics, Joseph equality, cross-covariance dimensions, multi-frame propagation, variance along unmeasured velocities, and forward/reverse consistency. PSD flooring/jitter must be logged as a numerical correction and not used to hide a mathematically invalid process or posterior. The trust-region damping is a solver device, not an unreported physical observation.

### 6. Correct physical interpretation of visibility and observability

- No object/background compositing objective. RGB-D observations come from the **actual measured foreground** and the transported canonical point-cloud render.
- Missing rendered support means *no predicted pixel value*. Record unsupported foreground coverage as a separately defined observation/identifiability issue, not a fake RGB-D residual.
- Neighboring points can share pixels. Preserve their coupling in the image Jacobian and compare hypothesis renders of the **whole scene** where needed.
- Test derivative behavior of front-most selection, point occlusions, footprint overlap, soft visibility/band selection, and measure/production image resampling. Make any deliberately frozen contributor set explicit.
- At pixel scale, RGB/depth of a textureless surface can make slide along that surface locally unobservable. The Fisher/null directions must reflect this; do not turn weak evidence into high-confidence point motion by hand.
- For point `i` where image measurements provide no information, **the Kalman prediction still occurs with growing uncertainty**. A hold-last-measured trajectory used *only for coherence feature computation* must never overwrite the physical filtered state or claim that a measurement arrived.

### 7. Independent motion-coherence extractor — exact reference logic

After Phase 1 has saved each physical XYZ/velocity trajectory, its marginal uncertainty and **direct-image-information history**, build a separate coherence graph on each endpoint's **canonical geometry**. It must be independent of the **correction group size**.

Reproduce the reference default **192 representative nodes; symmetric fixed canonical kNN, k=12**. Use the source code's stable canonical grouping/centroid formation. Do not set the coherence graph count through the point count / 64, 128, or 256 IEKF occupancy.

Compute trajectory metric consistency, not displacement magnitude. For representatives `a,b`:

```
ell_ab(k) = ||x_a(k)-x_b(k)||
ellbar_ab = median_over_valid_times ell_ab(k)
z_ab(k) = log((ell_ab(k)+epsilon)/(ellbar_ab+epsilon))
C_ab = sqrt( weighted_mean_k z_ab(k)^2 )
```

Use the pinned reference's `candidate_hierarchy`, `edge_distributions`, complement-relative frame estimation, `cross_validated_rigidity` on **representative-node folds**, world-frame moving-side disambiguation, and `select_moving_component`; preserve candidate and ambiguity outputs. Do not use ground truth, motion thresholds, target moving fractions, manually seeded object classes, or camera holdouts.

**RGB-D-specific modification needing a derivation:** A point can move under process prediction while unobserved. Purely predicted trajectories must not create confident rigid-motion evidence. Compute representative trajectory evidence using the point's direct rendered-image information (e.g. per-frame observed support and covariance/information); **do not equate every sample of a predicted trajectory with a measured location**. A zero-order hold is a diagnostic feature representation, not a Kalman state correction: demonstrate whether it biases the edge metric. Compare rigorous uncertainty-weighted/information-supported temporal edge statistics against the reference metric in simulated occlusion tests. Use the variant that has a stated probabilistic interpretation and consistent selection, and record any deviations from the pinned source. Calibration must not use the blade ground-truth part labels.

Free motion support must remain high-precision and incomplete when evidence is weak; keep `STATIC_COMPATIBLE` and `INCOHERENT_OR_AMBIGUOUS` distinct. Freeze the selected high-confidence moving core before Phase 2.

### 8. Phase 2 — exactly RigEuler's **shared rigid SE(3) pose + twist IEKF**, now image-rendered points

Use the selected moving-core canonical coordinates `p_i`. At frame `k`, for a physical moving point:

```
x_i(k) = T_k p_i
```

The one-part filter has state `(T_k ∈ SE(3), xi_k ∈ R⁶)` with 12D **left-error** `[delta_tau(6), delta_xi(6)]`, where `xi` is the **spatial/world twist**. Initialize `T_k` and `xi_k` by RigEuler's `kabsch_initialize`; this is an initialization, **not the final estimator**. The canonical anchor has `T_anchor = I`.

Copy the exact reference process (`rigid_part_runtime.RigidState`):

```
E_k = Exp(dt * hat(xi_k))
T_(k+1)^- = E_k T_k^+
xi_(k+1)^- = xi_k^+
F_k = [[Ad_(E_k), dt J_l(dt xi_k)], [0, I6]]
P_(k+1)^- = F_k P_k^+ F_kᵀ + Q_k
```

Use the reference 12D white-acceleration noise, including `Q_pose,twist ∝ dt²/2` and `Q_pose,pose ∝ dt³/3`. Define prior error in the same left chart:

```
e_tau = Log(T (T_minus)^-1)^vee
e_xi = xi - xi_minus
```

and its **actual** local retraction Jacobian `J_e`. On each IEKF iteration, render the entire colored point cloud under the shared moving-core transform while stationary points remain under identity; use the **same** all-camera RGB-D observation model as Phase 1. The rigid measurement Jacobian is chained from point XYZ through `x_i = T p_i`:

```
left delta_tau = [delta_rho, delta_phi]
delta x_i = delta_rho + delta_phi × (T p_i)
H_rigid = H_xyz * J_(point position wrt left SE3 pose)
```

**No direct image derivative with respect to twist**; twist is learned from pose-twist process covariance. The small 12D IEKF solves the exact reference fixed-prior correction (with `J_e` for the nonlinear Lie prior), retractions, and `12×12` final posterior information; transport/reset covariance into the new tangent chart consistently, without a second reset or replacing P with its initial value.

For fidelity, use the reference's `run_rigid_window` logic: predict, multiview render, correction, accept/reject by *same* image likelihood and fixed prior, store `T`, `xi`, `P_tau`, diagnostics in **original chronological frame indexing** irrespective of processing direction. Keep spatial pose conventions consistent and verify `T_(k+1) T_k^-1` versus `T_k^-1 T_(k+1)` with numerical tests.

Do not introduce a per-frame free 6DOF pose optimizer as a substitute for this recursive shared rigid IEKF. It may exist as a diagnostic ablation only.

### 9. Joint estimation: exact post-hoc trajectory reduction; only then image refinement

After all rigid poses are estimated, run the actual vendored or directly referenced `joint_reduction/rigid_trajectory_reduction.py::reduce_trajectory`. Copy exact Lie algebra conventions and helper functions (including nested imports) with test equivalence to the pinned reference; do not rewrite its finite rotation geometry from memory.

Use the **original chronological** rigid trajectory. Anchor the appropriate endpoint and transform into that endpoint gauge before reduction. The spatial relative increments are

```
Delta_T_k = T_(k+1) T_k^-1
zeta_k = Log(Delta_T_k)^vee / dt_k
```

Normalize translational twist components by the moving-part geometric scale `L`, then compute weighted SVD and `gamma_1dof` (a diagnostic, **not** an independent joint classifier). Compute:

- **Prismatic:** fixed unit direction `u`, `R(q)=I`, `t(q)=q u`, sequential signed physical displacements.
- **Revolute:** fixed unit axis `u` and finite pivot point `a`, `R(q)=Exp(q[u]×)`, `t(q)=(I-R(q))a`; pivot line constrained to a consistent axis gauge near the part; unwrap angle sequence; detect small-angle/far-axis mimicry of translation as an identifiability degeneracy, *not* an arbitrary label threshold.

The reduction code must report one-DOF rank, eigenvalues, finite-axis singular values/conditioning, per-frame left-chart pose residuals, RMS translation and rotation, and displacement versus stationary model. **Do not pass RGB-D images into this reduction**; RigEuler intentionally determines the initial joint from physical rigid trajectory first.

Next, run image-based **fixed-geometry joint IEKF** using the **same rendered RGB-D measurement operator**: for the selected candidate, each frame's dynamic joint state is `(q, qdot)` with continuous-time state transition and a correctly propagated `2×2` covariance. Joint *axis geometry* is one constant quantity for the whole sequence, never a newly estimated per-frame parameter. If geometry is refined after initialization, that is a **single shared global/static parameter** fitted using all frames with a properly represented global covariance—not 60 independently moving axes.

Compare prismatic and revolute candidates with the **same complete** multiview measurements, with defined statistical assumptions, identifiable parameter counts and appropriate degeneracy handling. Keep a third `NO_IDENTIFIABLE_JOINT` result. Do **not** select based on one arbitrary ratio or a threshold created because one synthetic static test failed. In particular, investigate the previous `3 × median footprint` excitation gate: it was retrofitted after a false static-scene detection and cannot be treated as a proven property of the Kalman model. Treat physical excitation using measured information and proper model comparison, reporting ambiguity when insufficient.

Rigid and joint stages must never read simulation part IDs/joint type/ground truth during inference. Test both revolute and prismatic under translation-only, rotation-only, mixed/static, and sparse/noisy evidence.

### 10. Forward/backward and endpoint gauge

Run physically independent first- and last-endpoint filters (all cameras), each using its own canonical point set and saved full original-time pose/point/covariance histories. The backward canonical point cloud is the observed final endpoint, **not** `state[0]` after chronological reindexing. Record anchor index, processing order, original-time indexing, signed velocity, camera calibration and time provenance.

Transform each selected joint into the same calibrated world frame, match axis up to sign (`u ≡ -u`, `q ≡ -q`) and compare **absolute world line geometry** for revolute or world direction for prismatic plus chronological `q_k` after consistent zero-gauge. **Never compare endpoint point IDs across directions.** Do not multiply correlated forward/backward posterior densities based on the same camera frames. If a combined model is needed, optimize a single shared model on **unique** raw RGB-D images with endpoint priors handled only once; comparisons should remain as diagnostics if joint not yet identifiable.

### 11. Support completion: image-rendered hypothesis test, not arbitrary per-point labels

Only after reliable joint geometry, for each existing endpoint point `i` anchored at `a_i`, compare

```
H_static: x_i(k) = p_i
H_moving: x_i(k) = T_J(k) T_J(a_i)^-1 p_i
```

**Both hypotheses render a complete RGB-D scene**, including current assignments of other points; pixel contributions, overlap, depth ordering, mask validity, and missed coverage create coupling. Do not treat each point's independent `DeltaE` as an exact posterior probability when points share pixel footprints. A single-point fast path must be verified against **full-scene rerender** at multiple frames and after other label changes; no hidden dependence on a previously fitted occlusion configuration.

For a rigorous probabilistic classifier, define a global latent label vector `z` and state its observation likelihood `p(Y | z, T_J)` and any prior `p(z)` clearly. Use an inference approximation only if it reproduces **exact enumeration on small coupled-point scenes** and reports the approximation and ambiguity. Never identify a point as moving only because a same-colored overlapping static point would also render a similar pixel. Preserve `STATIC_COMPATIBLE`, `MOVING_JOINT_COMPATIBLE`, `AMBIGUOUS`, and the original high-confidence `MOVING_CORE`; report coverage and uncertainty of each decision. When static and moving are observationally indistinguishable, **retain ambiguity**, not a fabricated label.

Do not hallucinate missing points or initialize new ones as part of this phase. The task is labeling and transporting **existing** forward and backward endpoint points; geometric support completion from the endpoints must be explicitly separated from insertion/discovery of new geometry.

### 12. Exact end-to-end execution plan

**Stage A — Source and math audit:** Finish `reports/theory_to_code_audit.md`, including all Jacobian conventions, observation weights, covariance representations, and source function mapping.

**Stage B — Renderer and measurement:** Implement/repair pixel-faithful colored-point RGB-D forward/JVP/VJP, visible-front logic and coherent noise/coverage semantics. Gate tests before running any filter.

**Stage C — Phase-1 IEKF:** Run prediction and a *single* iterated correction per frame, all cameras, correct grouped basis, fixed prior and posterior. Dense-oracle on tiny scenes; full 60 frames only after numerical gates.

**Stage D — Coherence:** Separate fixed-count 192-node graph and validated observation-supported trajectories, extraction exactly as pinned reference (document each necessary point-specific adaptation).

**Stage E — Rigid Phase 2:** Implement `SE(3) × R6` IEKF, 12×12 prior, Lie retraction/reset, full posterior and all-camera image updates, retaining one frozen physical moving core.

**Stage F — Joint reduction/refinement:** Pinned reduction routines, physically fixed P/R geometry, image-IEKF refinement, identifiable selection.

**Stage G — World-frame forward/backward comparison:** No bogus point-ID matching or posterior multiplication.

**Stage H — Support completion:** Calibrated full-image candidate comparisons, overlapping-point dependence and explicit ambiguity. Test on independent scenes.

**Stage I — Complete runs, view and reporting:** Run all 60 frames and 8 cameras on `blade_103706` forward and backward; at least one distinct revolute synthetic scene and a static control; if feasible run an independent Sapien object with different geometry, without tuning on the test ground truth. Export observed RGB-D, predicted RGB-D, residuals/validity/coverage, per-frame IEKF innovations, covariance spectra, representative graph, Kabsch/rigid/joint trajectories, P/R candidate axes and support assignments. Viewer must show actual renderer/observation comparisons.

### 13. Nonnegotiable mathematical tests (not just functions returning `COMPLETE`)

**Renderer gates** (CPU + CUDA): `render` determinism; calibrated cameras; metric depth; coverage; edge/occlusion behavior; finite-difference `Jv` vs actual nonlinear perturbation, `Jᵀz` adjoint identity, multi-point overlap, visibility ordering transitions, all eight cameras simultaneously, no background residual.

**Kalman gates**: analytic `F,Q` vs a dense reference; fixed-prior IEKF iteration matches dense algebra; equivalent K formulation; strict one-assimilation-per-frame invariant; `R` scaling affects gain and covariance as predicted; unmeasured velocity receives information only through covariance; grouped tangent `Gᵀ` adjoint and dense restricted solve; Joseph covariance vs dense 6N oracle, including off-group image coupling and at least 5 steps; PSD and PSD-reset/chart-consistency.

**Coherence gates**: stationary scene, pure global frame drift, one relative rigid moving part, occluded-moving part, nonrigid random drift, forward/reverse frames, coherence independent of computational group occupancy, moving core validated by node-wise held-out prediction. GT never used for production selection.

**Rigid/joint gates**: exact relative transform and left-log convention; finite-difference `F` and `J_e`; zero-velocity initialization recovering revolute/prismatic motion from the *rendered images*; rank-1 twist diagnostics, prismatic translation, finite revolute pivot, degenerate far-axis translation, poor-excitation `UNRESOLVED`; backward axis sign and world gauge.

**Support gates**: full-rerender vs incremental score at multiple overlapping candidates; brute-force exact-label posterior on a small scene; static, moving, ambiguous outcomes with known visibility, textureless unidentifiable surface; no optimistic independent-pixel/point confidence.

**Ablations**: correct vs stale renderer Jacobian; full multiview vs a single camera test-only; exact small dense covariance vs approximation; reference fixed-count coherence vs occupancy graph; image IEKF shared rigid vs independently optimized poses. Record results; don't make the ablation the production algorithm.

### 14. Success and failure reporting

The earlier reported `blade_103706` trial (prismatic axis errors 0.81°/1.18°, q RMSE 2.4/5.6 mm, support moving-label accuracy ~15–16% among GT-labeled points, 20% covariance variance bias in one dense case) is the **baseline to challenge**, not success proof or an excuse to preserve its unverified approximations. These results were reported by the earlier run; their complete source edits were uncommitted and have not been independently verified.

A **rigorous implementation** may improve, match, or initially worsen the apparent blade metric while correcting a mathematical error. Never tune to keep a favorable number or to make a status green. Publish failures explicitly with the earliest failing gate and a numerical diagnosis. Separate:

- `FILTER_NUMERICALLY_VALID`: source-equivalent recursive IEKF, Jacobians and covariance tests pass.
- `RIGID_MOTION_VALIDATED`: one physical rigid body and consistent time trajectory supported.
- `JOINT_IDENTIFIED`: identifiable P/R geometry and trajectories with documented evidence.
- `SUPPORT_VALIDATED`: new moving/static labels meaningful on independent scenes, not just a complete run.

Do not call the whole reconstruction scientifically complete unless all four requirements are met. Generate `reports/rigeuler_rgbd_theory_validation.md`, `reports/rigeuler_rgbd_test_matrix.csv`, machine-readable per-frame diagnostics and final viewer output. Make code and report changes reviewable. After a successful end-to-end validation, commit source, tests and reports together with explicit notes on approximations and residual weaknesses (do not commit gigantic SSD outputs).

### 15. Command to Claude Code

**Implement this contract end to end.** You have authority to **replace** mathematically incorrect existing modules, even if they passed a previous experiment. Reuse only equations/functions that you have checked against the pinned source and dense/finite-difference oracles. Do not switch to an association-based optimizer, hand-built loss-only fitter, a persistent rigid-supernode model, or an invented joint classifier. Do not ask the user to choose between multiple algorithmic redesigns; execute the specified RigEuler methodology adapted to colored XYZ+velocity points and rendered RGB-D observations. Finish with the actual source diff, exact commands, full test results, forward/backward quantitative joint and support outcomes, and a candid list of any remaining failures.
