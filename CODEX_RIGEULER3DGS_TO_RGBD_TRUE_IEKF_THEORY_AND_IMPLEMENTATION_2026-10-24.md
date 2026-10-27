# CODEX — RigEuler3DGS → RGBDiekf TRUE IMAGE-BASED IEKF (corrected Kalman theory)

**AUTHORITY / REPLACES PREVIOUS SPECIFICATION — includes corrected IEKF theory.** This document **supersedes** `CODEX_RIGEULER3DGS_TO_RGBD_FINAL_PORT_2026-10-24.md`, `CODEX_RIGEULER3DGS_TO_RGBD_IMAGE_IEKF_CORRECTED_2026-10-24.md` and every previous instruction that made point-to-point RGB-D associations the Phase-1 measurement model. The earlier document incorrectly substituted an association-based UVZ estimator for RigEuler's renderer-based image observation. Do not implement that substitution.

**Source:** `SamueleMara/RigEuler3DGS` pinned at `481b2cf2d5602932286948cf14114f9caee4de3c`. **Target:** `SamueleMara/RGBDiekf` reviewed `main` at `f913d5628d29fd6c4494aeea5fad83874962dc40`. Inspect the current local working tree/HEAD before changing it. Preserve existing results and baseline paths. **Data:** `blade_103706`, 60 frames, 8 synchronized calibrated RGB-D cameras, chronological timestamps, independent forward and backward endpoint catalogs.

## 0. The exact intended algorithm (read before writing code)

**IMPORTANT mathematical correction:** an RGB-D pixel residual loss alone is NOT an EKF or IEKF. Section 2 contains the binding state transition, predicted covariance, fixed-prior iterated Kalman equations and posterior. Those are requirements, not optional interpretation.

1. **Phase 1 — FREE motion.** Initialize a colored XYZ point cloud from real first-frame RGB-D (forward) or final-frame RGB-D (backward), using all cameras. Every point has its own `[world XYZ, world velocity]` state and uncertainty. At every next frame, **transport ALL colored points using the IEKF motion prediction; project/splat/rasterize that transported cloud into each of the 8 cameras, producing predicted full-frame RGB and metric depth images**. The next frame's **observed RGB and depth images** are the measurement. Apply the grouped *iterated extended Kalman update*: process prediction, fixed-prior information update with renderer H/Hᵀ, and posterior covariance after final convergence. Rerender and relinearize during IEKF iterations. Computational supernodes are correction bases ONLY, never physical rigid bodies.
2. **Coherence.** Extract a reliable, high-confidence moving core from the saved free point trajectories with the pinned RigEuler coherence method and an independent, fixed-count representative graph. No joint model or physical supernode splitting here.
3. **Phase 2 — ONE shared rigid body.** Initialize a shared `SE(3)` pose **and spatial twist** from coherent point trajectories. Run RigEuler's temporal rigid IEKF. Render the **whole scene** for each camera: confirmed moving-core points transformed by the common pose; remaining endpoint points at their stationary/identity reference positions for visibility. Compare predicted RGB and depth **images** with the actual frame images. Optimize ONE moving-body pose/twist, not separate supernode poses.
4. **Joint.** From optimized free rigid poses, run the pinned `rigid_trajectory_reduction.reduce_trajectory`: relative twists, dimensionless rank-one `gamma_1dof`, PRISMATIC and REVOLUTE pose-space fits and residuals. Only then optionally refine each fixed-joint hypothesis using the **same rendered-image RGB-D observation objective**. Do not select a joint using point-to-point association residuals.
5. **Bidirectional.** Compare forward/backward joint geometry and `q(t)` in chronological world coordinates (separate canonical catalogs; no matched point IDs required). Fuse compatible support/evidence without double-counting pixels.
6. **Support completion.** Given a validated joint, revisit uncertain points in BOTH endpoint clouds. For each, compare the predicted image evidence under identity `I` versus anchor-relative joint transport `J(q_k)J(q_anchor)^{-1}` within the complete rendered scene across all observed frames. Mark static, moving or ambiguous; no new geometry yet.
7. **Debug.** Show observed RGB/depth/mask, rendered RGB/depth/coverage, RGB/depth residual maps, transported XYZ, computational and coherence supernodes, chosen core, Phase-2 rigid image render, BOTH joint-candidate image renders and axes, and forward/backward support. Confirm every image corresponds to the actual camera/frame, not a projected-point match table.

**Non-negotiable:** No dedicated camera train/selection/assessment split; all 8 cameras contribute to every production update. No Gaussian opacity renderer, alpha compositing, artificial background, black-background loss, GT joint labels, ground-truth part masks, hard-coded object-specific tricks, physical-supernode splitting, K-joint inference, or point insertion. **A simple point-cloud RGB-D rasterizer IS required; the prohibition is on extraneous background and 3DGS rendering, not on predicted image rendering.**

## 1. Verify what RigEuler actually optimizes — inspect these exact functions

Read and map call sites before editing:

- `RigEuler3DGS/estimator/renderer_bridge.py`: `build_linearization`, `FrameLinearization.predicted`, `apply_H`, `apply_HT`, `build_block_fisher`. This is the **measurement-operator interface** and differentiates per-camera **rendered RGB and depth**, not individual point-to-point associations.
- `RigEuler3DGS/estimator/sparse_iekf.py`: `_frame_terms`, `measurement_operators`, `iekf_frame_step`. Specifically study the repeated **render → pixel image residual → H/Hᵀ → grouped matrix-free normal solve → true-image objective line search → rerender** loop. The normal matrix retains cross-point co-visibility coupling through `H`.
- `RigEuler3DGS/estimator/supernodes.py`: `GroupedObservation.expand`, `reduce`, `apply_H`, `apply_HT` and grouping bases. They parameterize the correction; they do **not** define physical rigid part trajectories.
- `RigEuler3DGS/tools/run_motion_coherence_support.py`: distinct `--iekf-supernode-size=64` and `--supernode-count=192`; fixed canonical representative graph.
- `RigEuler3DGS/estimator/motion_coherence_support.py`, `estimator/coherence_extraction.py`: log-distance coherence, candidate hierarchy, held-out **representative nodes**, complement-relative Kabsch, candidate selection.
- `RigEuler3DGS/estimator/rigid_part_runtime.py`: `RigidState`, `kabsch_initialize`, `process_matrix`, `process_noise`, `run_rigid_window`. This is one **shared pose/twist temporal IEKF** after moving-core discovery.
- `RigEuler3DGS/joint_reduction/rigid_trajectory_reduction.py`: `relative_increments`, `one_dof_diagnostics`, `reduce_trajectory`; pose-space models before sensor refinement.

**Port the algorithmic interface, not the Gaussian model.** Here the primitive is a measured colored 3D sample/surfel with canonical world XYZ, persistent RGB, source pixel/camera geometry and (where available) a projected surface footprint. Its measurement is the **predicted RGB/depth image**. 3DGS alpha, appearance optimization, SH rotations, opacity, random background and Gaussian-scale training do not transfer.

## 2. THE FILTER THEORY — image-based **iterated extended Kalman filter**, not plain image fitting

**This section is normative and replaces the earlier standalone `E_image` description.** RigEuler's `sparse_iekf.iekf_frame_step` is a Bayesian **predict → fixed prior → iterated nonlinear measurement update → once-per-frame posterior covariance** procedure. Its measurement function is a renderer. Port exactly that structure, changing only (i) the state from 12D Gaussian motion to 6D point motion, and (ii) the nonlinear renderer from Gaussian splats to RGB-D colored-point rasterization. See pinned `sparse_iekf.py` lines 425–612, 619–738 and 990–1025, and `renderer_bridge.py` lines 70–122.

### 2.1 State, process prediction and covariance prediction — ONCE per frame

For **each point** `i`, state `s_i=[x_i; v_i] ∈ R⁶`; all `N` states stack to `s ∈ R^(6N)`. The canonical 3D point and fixed color originate in measured RGB-D at the genuine endpoint. Each processing direction has its **own** canonical point catalog, its own state, and its own covariance. Supernode groups do not define physical rigid parts.

For chronological index `k`, from the last accepted posterior `(s⁺_previous, P⁺_previous)`:

```text
s⁻_k = f_(Δt)(s⁺_previous)
F_k  = ∂f/∂s evaluated at s⁺_previous
P⁻_k = F_k P⁺_previous F_kᵀ + Q_k
```

Under the existing constant-velocity model per point:

```text
[x⁻]   [I₃  Δt I₃] [x⁺]
[v⁻] = [ 0     I₃] [v⁺]
```

Use real **signed** `Δt`: positive for forward and negative for backward processing. Retain the existing physically dimensioned process covariance; for continuous white velocity acceleration covariance `Q_a` in reverse or forward time, its `[x;v]` sub-blocks have magnitudes `|Δt|³/3`, `Δt|Δt|/2`, `|Δt|`, respectively (plus any separately declared direct-position diffusion `Q_x |Δt|`). Verify the sign and PSD rather than using a positive-Δt shortcut for backward prediction.

The predicted mean and **predicted covariance are held fixed over ALL IEKF inner iterations for frame `k`**. Never propagate the process again inside the inner loop. Never reset `P⁺` to the initial endpoint covariance at the next frame.

### 2.2 The actual measurement `y_k = h_k(s_k) + n_k`

The observed data at frame `k` are the **actual** synchronized RGB image `I^obs_{c,k}` and metric depth image `D^obs_{c,k}` for every camera `c=0,...,7`, with measured foreground/depth-valid masks, calibrated camera matrices and declared measurement noise.

The nonlinear predicted measurement is the **RGB-D image rendered from the transported colored point cloud**:

```text
h_(c,k)(s_k) = [ vec(Î_(c,k)( {x_i,k}, {fixed RGB_i} ));
                 vec(D̂_(c,k)( {x_i,k}, {fixed RGB_i} )) ]
y_k          = stack_c [ vec(I^obs_(c,k)); vec(D^obs_(c,k)) ]
y_k          = h_k(s_k) + n_k,    E[n_k]=0,   Cov(n_k)=R_k.
```

`h` depends on **point positions** and predicted visibility; instantaneous image values do **not** depend directly on instantaneous velocity, so the measurement Jacobian has **zero instantaneous velocity columns**. The process-model pose/velocity cross covariance is how image evidence updates the velocity. There are no per-point orientation states.

Use valid observed foreground pixels and predicted coverage/visibility to define which residual channels are well defined, with zero precision for invalid depth. Do not read background RGB/depth as valid object measurements; do not fill missing predictions with fictitious zero RGB/depth; and do not let a trial minimize its error simply by losing previously covered observed pixels. Freeze a fair observation domain during trial acceptance or explicitly reject unsupported coverage loss under an audited validity rule. Record the active support and any such rejection. The active-domain/occlusion rule is part of the measurement model and must be **the same** in energy, gradient, normal operator and posterior information.

`R_k` is the measurement covariance **in image space**, dimensioned for RGB and depth. You may use declared per-pixel/channel diagonal or block-diagonal noise for efficient weighted images. This conditional pixel independence is an assumption, not an implication of shared point contributors. If robust IRLS weights are used, freeze them within each linearized solve and label the approximation rather than asserting an exact Gaussian posterior.

### 2.3 The measured RGB-D likelihood is only ONE TERM of the fixed-prior posterior

At frame `k` write:

```text
e_j = s_k^(j) − s⁻_k                      # Euclidean XYZ/velocity chart
r_j = y_k − h_k(s_k^(j))
H_j = ∂h_k/∂s |_(s_k^(j))               # includes ALL cameras and their pixels
Λ⁻  = (P⁻_k)^−1
W_j = R_k^−1 restricted to the active valid measurement channels
```

The **IEKF target**, up to a normalization constant and subject to the selected measurement support, is:

```text
L_k(s) = ½(s − s⁻_k)ᵀ Λ⁻ (s − s⁻_k)
       + ½(y_k − h_k(s))ᵀ W_k (y_k − h_k(s)).
```

This is the **fixed-prior negative log posterior** for a Gaussian predicted state and locally Gaussian image observations—not a free-standing RGB-D least-squares fit. State/process covariance and the actual sensor covariance are essential. The prior term is **NOT** rebuilt around the previous inner iteration.

At iteration `j`, use `h(s_j + δ) ≈ h(s_j) + H_j δ`. The linearized **full-state** normal equation is:

```text
(Λ⁻ + H_jᵀ W_j H_j) δ = H_jᵀ W_j r_j − Λ⁻ e_j.
```

For full, unrestricted Euclidean states and a fixed Gaussian `R`, this is algebraically the iterated Kalman update:

```text
K_j = P⁻ H_jᵀ (H_j P⁻ H_jᵀ + R)^−1,
s_(j+1) = s⁻ + K_j [ y − h(s_j) + H_j(s_j − s⁻) ].
```

**Both formulas must be present in implementation documentation.** The first is what RigEuler efficiently solves in information/operator form, avoiding the huge image-space innovation inverse. The second makes explicit that the algorithm is an **iterated extended Kalman filter**, not an unconstrained renderer fit. With nonlinear rendering, damping/trust region and visibility switching, they describe a local/Gauss–Newton IEKF approximation; do not imply the full nonlinear posterior is exactly Gaussian.

### 2.4 Computational supernodes restrict UPDATE increments; they do not impose permanent rigidity

Use the fixed canonical partition and RigEuler's correction expansion map `G`. For each point in computational supernode `g`, with its canonical offset `r_i = p_i − c_g`, the tangent map in `[XYZ;velocity]` coordinates is:

```text
δx_i = δt_g − [r_i]× δθ_g
δv_i = δu_g − [r_i]× δω_g
δs = G δζ,
```

where `δζ_g` has 12 components (6 for position-correction field; 6 for velocity-correction field); **the point still only has its own 6D `[XYZ;velocity]` state**. If the existing implementation uses a normalized `G` that spans the same structural range, preserve it. The canonical membership and correction basis are kept fixed; point means evolve freely from frame to frame.

Substituting `δs = G δζ` into the preceding **IEKF** linearization gives the **actual grouped filter information update**:

```text
A_g δζ = b_g
A_g(ζ) = Gᵀ [ Λ⁻(Gζ) + H_jᵀ(W_j H_j(Gζ)) ]
b_g    = Gᵀ [ H_jᵀ(W_j r_j) − Λ⁻ e_j ].
```

Implement these as **matrix-free operators**, and solve them with appropriately conditioned grouped PCG and trust-region/line-search control. Do not approximate `Hᵀ W H` as a sum of independent per-point terms: multiple points and computational groups contribute to the same rendered image pixel. The **full pixel coupling is preserved in the MEAN update** through `H`/`Hᵀ` even if the saved posterior is approximated.

The current accepted iterate is corrected `s_(j+1) = s_j + G δζ` (with the chosen globalization rule). Recompute the true full rendered image and prior energy for trial acceptance. If accepted, rerender and relinearize. Do NOT make the individual points follow a rigid `T_supernode(k)` trajectory over time.

### 2.5 Renderer, pixel Jacobian, matrix-free adjoint (the MEASUREMENT OPERATOR)

Implement a **minimal differentiable colored point/surfel rasterizer**. This replaces the Gaussian-specific `renderer_bridge` internals while providing the same interface: `predicted`, `apply_H`, `apply_HT`, and block Fisher for preconditioning/posterior approximation. It is **not** a Gaussian-opacity/background renderer and not a nearest-neighbor raw XYZ association.

For a point in camera `c`, project via `K_c`, `T_cw` to pixel `(u_i,v_i)` and positive metric depth `z_i`. Give the fixed colored surface samples a nonlearned finite compact footprint derived from the original RGB-D sampling and calibrated geometry, and maintain deterministic visible/front-surface contributor lists. Within a **frozen** support and visibility ordering at a given IEKF iteration, normalized splatting can be defined by:

```text
W_p     = Σ_i w_ip
Î_p     = (Σ_i w_ip color_i)/W_p
D̂_p     = (Σ_i w_ip z_i)/W_p,
coverage_p = [W_p > 0].
```

Differentiate the **actual** chosen footprint. If its radius/shape depends on depth, include that derivative too (or freeze its size consistently within the current linearization and disclose the approximation). For fixed colors and fixed footprint-size within the linearization:

```text
d Î_p = (1/W_p) Σ_i (color_i − Î_p) d w_ip,
d D̂_p = (1/W_p) Σ_i [ (z_i − D̂_p) d w_ip + w_ip d z_i ],
d[u_i,v_i,z_i] = J_(projection,c)(x_i) dx_i.
```

**RGB gradients come from transported projected point footprints and their color variation**; they are not zero merely because per-point colors are fixed. Where surface texture is uniform, RGB information about in-plane motion may genuinely be weak. Do not hallucinate an image gradient.

The image-space Jacobian is `H = [∂rendered_RGBD/∂x, 0_v]`; it includes contributions of **all cameras**. Implement `H(v)` and `Hᵀ(z)` as the correct scatter/gather adjoints over the SAME frozen contributor lists. `Hᵀ` sums each camera's image adjoint into each point state. Validate (i) finite-difference JVP against rerenders while support is stable, (ii) `<H v,z>=<v,Hᵀ z>` with stringent numerical tolerance, (iii) projected-depth units, and (iv) nonzero RGB image sensitivity for textured point motion.

Do not use 3DGS Gaussian alpha, learned scale/opacity, SH or artificial background. Depth ordering/front-surface visibility is geometric point rasterization, not background rendering. Distinct source samples do not constitute independent observed image pixels; the likelihood is indexed by **camera and observed pixel**.

### 2.6 Posterior COVARIANCE is computed ONCE from the final accepted linearization

After the last accepted iterate `ŝ⁺_k`, compute the final image Jacobian `H_*` and weights `W_*`. **The gain is not re-applied to the mean after every inner iteration as if each rerender were an independent new sensor measurement.** The IEKF represents ONE assimilation of frame `k`.

For the chosen **restricted** linear correction, with full predicted covariance if represented, the linearized estimator gain is:

```text
A_g = Gᵀ (Λ⁻ + H_*ᵀ W_* H_*) G
K_G = G A_g^−1 Gᵀ H_*ᵀ W_* .
```

The covariance of **that linearized restricted estimator** (if exact full `P⁻` and the above `K_G` were available) is the Joseph expression:

```text
P⁺ = (I − K_G H_*) P⁻ (I − K_G H_*)ᵀ + K_G R K_Gᵀ.
```

It is NOT generally `G A_g^−1 Gᵀ`, which describes curvature in the restricted-coordinate subspace and omits unconstrained prior uncertainty. For **unrestricted** classical EKF, the information-form covariance is `(Λ⁻ + H_*ᵀ W_* H_*)^−1` under the linearized full-state Gaussian assumptions.

**What the pinned RigEuler source ACTUALLY does:** `sparse_iekf.py` ends with `approximate_posterior_information(Lambda_prior, fisher_pose)` then inverts its **per-Gaussian blocks**. This is an explicit blockwise posterior approximation and is NOT the full global covariance implied by cross-Gaussian image coupling. Do not describe RigEuler as computing the exact global Joseph covariance. Reproduce its practical contract for the XYZ-only port if needed: a correctly dimensioned **per-point 6×6 information/Fisher approximation**, plus clear provenance that cross-point/cross-group posterior correlations are omitted (or use a verified scalable covariance operator if one is already implemented). This need not block the global matrix-free MEAN solve.

The old `grouped_covariance.py` closed rank-24 **independent-computational-group** claim relies on per-point UVZ factors, and is not exact for rendered-image factors coupling distinct groups. Do not silently transplant that exactness claim. Retain the baseline source unchanged. On small dense image cases, compare: (a) true full global correction and posterior, (b) chosen grouped correction, (c) exact restricted Joseph covariance and (d) production declared block approximation; quantify errors explicitly. Save SPD/conditioning/observability diagnostics. Point XYZ is Euclidean and needs no Lie covariance chart reset; **Phase 2's shared `SE(3)` pose DOES require its appropriate Lie-group covariance reset/transport after accepted retraction**.

### 2.7 Exact per-frame IEKF flow — mandatory pseudocode

```text
for anchor in [frame_0, frame_last]:
    initialize endpoint point catalog from measured RGB-D across all cameras
    initialize per-point state [XYZ, velocity], P_anchor and canonical correction groups

    for k in complete chronological order (forward) or reverse order (backward):
        if k == anchor: save canonical gauge and prior; continue
        Δt = timestamp[k] - timestamp[previous]

        # ONE process prediction this frame; these values remain fixed below
        s_minus = f_Δt(s_previous_plus)
        P_minus = F_Δt P_previous_plus F_Δt.T + Q_Δt
        Lambda_minus = inverse_or_linear_solve(P_minus)
        s_current = s_minus

        for j in range(max_iekf_iterations):
            render = renderer.render_full_rgb_depth(s_current.xyz, fixed_rgb, all_8_cameras)
            residual, W, valid_pixel_domain = observation_terms(
                actual_RGBD_images[k], render, masks[k], declared_noise)
            H = renderer.freeze_jvp_vjp(render.contributors, valid_pixel_domain)

            prior_error = s_current - s_minus
            b = GT(HT(W * residual) - Lambda_minus(prior_error))
            def A(z):
                dz = G(z)
                return GT(Lambda_minus(dz) + HT(W * H(dz)))

            dz = grouped_PCG(A, b, proper_preconditioner)
            s_trial = s_current + G(dz)   # appropriately scaled trust/line-search step
            trial_image = renderer.RENDER_AGAIN(s_trial.xyz, fixed_rgb, all_8_cameras)
            accept iff TRUE nonlinear [fixed-prior + image] objective improves
                on a fair, auditable measurement domain
            if accepted: s_current = s_trial
            if converged_or_stalled: break

        # ONE measurement assimilation / posterior covariance per video frame
        render_final = renderer.render_full_rgb_depth(s_current.xyz, fixed_rgb, all_8_cameras)
        H_final, W_final = renderer.linearize_final(render_final, actual_RGBD_images[k])
        P_plus = declared_posterior_covariance_method(P_minus,H_final,W_final,G)
        save s_plus=s_current, P_plus, per-camera images, coverage,
             residuals, innovation/Fisher, accepted/rejected iterations
```

**Strict failures to prevent:** (1) an image-loss optimizer with no `s_minus`, `P_minus` or prior information; (2) reinitializing the prior inside the inner loop; (3) applying multiple covariance updates for repeated render linearizations of the SAME frame; (4) replacing `HᵀWH` with independent per-point UVZ Hessians; (5) using `associate()` for the production observation; (6) treating the single per-frame image residual loss as the entire EKF formulation; (7) silently discarding all RGB Jacobian terms.

## 3. Phase 1 source changes: free XYZ IEKF with rendered-image updates

Target changes in `pointwise_rgbd/`:

1. **Add** `rgbd_point_renderer.py` (or `render_rgbd_points.py`) with stable API: `render(points_xyz, fixed_rgb, source_footprints, cameras, image_size, mask_metadata) -> per-camera predicted RGB, depth, coverage, frozen_contributor_cache`; `apply_H`; `apply_HT`; optional diagonal/block Fisher preconditioner. GPU and CPU oracle must agree.
2. **Modify** `free_phase1.py`: keep endpoint initialization, real/signed timestamps, `[XYZ,velocity]` process model, canonical computational grouping, independent first/last catalogs, all-camera loading and point state export. **REMOVE `associate()`/`pack_factors()` as the production Phase-1 measurement objective.** Replace with render/relinearize/image pixel residuals. Source endpoint pixels remain endpoint geometry priors exactly once; *new* frame images supply new measurements. Do not change canonical RGB when making a trial update.
3. **Modify** `grouped_point_iekf.py` to accept renderer `apply_H`/`apply_HT` and implement the same grouped `G`, `Gᵀ` matrix-free image-information solve as RigEuler. Preserve free per-point outputs; no persistent supernode poses.
4. **Modify** covariance code only where required by pixel co-visibility as in §2.3; baseline remains separate, testable and unchanged. Validate CPU/CUDA, true objective decreases and observability on real image pixels.
5. **Do not call `association.associate` in the production IEKF update.** It may remain for isolated legacy tests, diagnostics or point-cloud comparison, **not as the measurement equation or optimizer's primary evidence**. The rendered RGB and depth images are the actual predictions and the next camera images are the actual observations.
6. Save per-frame/per-camera **predicted RGB, depth, coverage**, image-space RGB/depth residual maps, per-camera image likelihood, contributor counts, image JVP/VJP diagnostics, visible/cropped point counts, point-position/velocity trajectories and covariance diagnostics. Replace historical `factors.sqlite` semantics with versioned **image measurement operator** provenance; do not pretend prior matched raw point factors remain valid for the new model.
7. **Invalidate old Phase-1 caches** as inputs to this image-model IEKF. Previous all-camera per-point trajectories used matched UVZ points, so their mean/covariance are different; they may be retained as **legacy association-baseline comparisons**, not silently reused. This corrects the earlier spec's instruction to reuse their Phase-1 trajectories as though the measurement model were identical.

### Normative IEKF algorithm

**Implement section 2.7 exactly**, including the once-per-frame process prediction, fixed prior, repeated image-space linearization and once-per-frame posterior covariance. The earlier pseudocode that merely said `grouped_PCG_map_update` without displaying the fixed prior and full Bayesian update was incomplete and is superseded.

## 4. Coherence stays the reference trajectory method

Use the pinned reference code, with **separate coherence representatives** (`--supernode-count=192`, `k=12`) and image-IEKF output trajectories. Fixed canonical point membership; symmetric canonical kNN edges; whole-sequence median log-distance strain; graph persistence, complement-relative and world-frame held-out **NODE** cross-validation; high-confidence dynamic core. Do not classify physical parts during Phase 1 and do not fit a joint to select the core. Report candidate rigidity, static-reference compatibility and **actual image-render residual diagnostics** so predicted-only drift is not mistaken for reliable measurements.

Important: coherence can be ported/tested independently on synthetic point trajectories while the new image Phase 1 is implemented; but **the previous association-based real trajectories are not evidence of the repaired image pipeline**. Real coherent support must be extracted again from its new rendered-image-based IEKF trajectories.

## 5. Shared rigid Phase 2 uses THE SAME image renderer

Port `estimator/rigid_part_runtime.RigidState` and `run_rigid_window`: ONE state `(T_k∈SE(3), ξ_k∈R⁶)`, left pose increments, correctly signed/chronological process `T⁻_{k+1}=Exp(dt ξ_k^) T_k`, process covariance from the reference, Kabsch initialization of **only coherent moving core**, spatial `ξ = Log(T_{k+1}T_k⁻¹)/dt`, actual endpoint gauge `T_anchor=I`.

**This stage is also an IEKF, not a per-frame SE(3) image least-squares solve.** Use the predicted state and its 12×12 covariance:

```text
E = Exp(Δt ξ_prev^)  [in SE(3)]
T⁻_k = E T⁺_previous
ξ⁻_k = ξ⁺_previous
F = [[ Ad_E,  Δt J_l(Δt ξ_prev) ],
     [   0 ,           I₆         ]]
P⁻_rigid = F P⁺_rigid Fᵀ + Q_rigid(Δt).
```

The RigEuler `rigid_part_runtime.py` code supplies the SE(3) adjoint, left Jacobian, white-acceleration/twist process noise and spatial-twist convention. Preserve the actual endpoint gauge; the backward run must use a consistent reverse-time transition. Its 12D error is `[δτ; δξ]`: **6 rigid pose tangent components plus 6 spatial-twist components, for the ONE moving body**, not per endpoint point and not per computational supernode.

At inner iteration `j` evaluate `e_j = [Log(T_j (T⁻_k)^−1); ξ_j−ξ⁻_k]` in the same **left** error coordinates used for the covariance, render the complete colored point scene, and linearize predicted pixels. For moving point `i`, `δx_i = [I₃, −[x_i]×] δτ`; static hypothesis points have zero rigid-state Jacobian. The image `H_rigid` has no direct twist derivative: `H_rigid=[H_image_xyz J_rigid, 0]`. The rigid measurement update must use the **Jacobian of the Lie-group prior error**, precisely as `joint_iekf._FramePrior.gradient_parts` does. Set `J_e,j = ∂e(T_j ⊞ δa)/∂δa|₀` in the chosen left chart (use an analytic expression or the 12D finite-difference reference). Its normal equation is:

```text
A_j = J_e,jᵀ Λ⁻_rigid J_e,j + H_rigid,jᵀ W_j H_rigid,j
b_j = H_rigid,jᵀ W_j [y−h(T_j)] − J_e,jᵀ Λ⁻_rigid e_j
A_j δa = b_j.
```

**Do not silently use `J_e=I` for finite SE(3) retractions**; that identity is exact in Phase-1 Euclidean `[XYZ,velocity]` coordinates but only a local approximation for Phase 2 when the current pose is away from the predicted pose.

Use the accepted Lie retraction `T_trial=Exp(δτ^) T_j`, `ξ_trial=ξ_j+δξ` and a fresh render to decide whether the **fixed-prior negative log posterior** decreases. Do not treat `δξ` as an instantaneous pixel-motion residual; it is inferred through pose/twist prior correlation. Form the **12×12 Phase-2 posterior** once at the final accepted linearization from the final `A_*` including `J_e,*ᵀ Λ⁻ J_e,*`, as RigEuler's `joint_iekf.frame_step` does, and transport/reset pose covariance consistently with the chosen left Lie chart; a reset is a **change of uncertainty coordinates**, not reinitialization. Implement and test the Jacobian via finite differences and a small dense 12D reference. The goal is a coherent temporally propagated free rigid trajectory, **before** fitting the one-DOF joint.

For each camera and frame, render the **whole endpoint catalog**:

- moving-core point XYZ `x_i(k) = T_k T_anchor⁻¹ p_i`, optimized by the common moving-body state;
- other canonical points at their world XYZ under **identity**, as the current static hypothesis, so they affect **point visibility/depth ordering** and scene pixels; they are **not** a photographic background.

Predict `Î_c,k` and `D̂_c,k` from the whole cloud and perform the **full predicted pose/twist IEKF measurement update** using the actual image observations, the fixed predicted state/covariance, the image Jacobian and posterior covariance update. The RGB-D residual is the likelihood component, never the entire filter objective. For moving points,

`δx_i = [I_3, −[x_i]×] δτ`, so `H_rigid = H_image_xyz * J_rigid`, and the twist enters via the predicted pose covariance/process, **not** as an imaginary instantaneous RGB derivative.

Use **all 8 cameras**. Re-render/relinearize per IEKF iteration. Retain endpoint source point uncertainty in a consistent MAP/Schur canonical estimate if optimization of canonical geometry is enabled; do not allow independently optimized per-frame SE(3) poses to replace the temporal IEKF without saying so. Save predicted RGB/depth and their image losses during Phase 2. Do not fit the revolute or prismatic axis yet.

## 6. Joint reduction — reference pose-space method, then IMAGE-model joint refinement

Run exact pinned `joint_reduction/rigid_trajectory_reduction.reduce_trajectory(R,p,canonical_xyz,members,P_tau,times)` on the Phase-2 poses, using the correct anchor gauge. Save:

- `ΔT_k = T_{k+1} T_k⁻¹`, `ζ_k = Log(ΔT_k)/dt`, part scale `L`;
- singular values and `gamma_1dof` of dimensionless twist stack (diagnostic, **not** a hard classifier);
- prismatic direction and `q_k`, revolute axis direction/point and `q_k`, finite-axis identifiability, full pose-space residuals, correct world and left-tangent covariance chart;
- free-rigid-vs-joint **3D point** discrepancy on the **same support**, before any new image fitting.

**Then** optionally refine each candidate in the image observation model by rendering its predicted moving points plus fixed static geometry into every camera. Model-specific image residual and actual coverage must be shown on the same observed image set. Preserve the original hypothesis's identifiability rules: a degenerate revolute candidate cannot veto valid prismatic translation, pure stationary/noise motion does not generate a joint, and genuinely indistinguishable hypotheses remain unresolved. Do not use arbitrary physical movement thresholds or make a joint pass by inflating RGB/depth noise.

## 7. Forward/backward and support completion use rendered-image evidence

Forward and backward have different canonical endpoint points, but all poses and joint axes use the same **calibrated world frame** and original chronological frame IDs. Compare type, axis up to sign, axis location up to axial gauge, and relative joint displacement gauge correctly. Combine compatible evidence without double-counting the same measured pixels as independent data.

For endpoint point `p_i` anchored at `a_i`, define two predicted trajectories:

```
x_static_i(k) = p_i
x_moving_i(k) = J(q_k) J(q_ai)⁻¹ p_i
```

Score both by their effect on **the actual predicted RGB and depth images** of the complete scene over all valid observed frames and all cameras. Occlusion must be computed from the projected point-cloud depth ordering. Only pixels meaningfully influenced/visible under the candidate hypothesis count; an unseen point cannot acquire confident membership. To avoid O(number_points × full_image_renders), use the renderer's cached local splat footprints/contributor lists for candidate image-change evaluations; check this fast path against actual complete rerenders on a small oracle scene. Do not return to exclusive point-to-raw-XYZ matching for support classification.

Use `STATIC_COMPATIBLE`, `MOVING_JOINT_COMPATIBLE`, `AMBIGUOUS` with no forced decision at the rotation axis, in occluded regions, or with insufficient image evidence. Complete **only existing forward/backward point catalogs**, no invented points.

## 8. Functional visual debugging — must match IMAGE measurements

Update `free_visualization.py`, viewer assets and output schema to display, separately for each frame and selected camera or all cameras:

1. Actual observed RGB, **metric** depth and valid foreground mask.
2. Rendered predicted RGB, depth and true point-cloud coverage for Phase 1.
3. RGB residual heatmap (per channel or norm), depth residual heatmap in **meters**, visible contributor/occlusion map, valid/invalid pixels and actual numerical per-camera loss.
4. Free transported 3D point cloud, observed/prediction-only status, velocity, covariance diagnostics and computational supernode colors.
5. Independent coherence representative graph and high-confidence moving-core labels, with point support counts.
6. Rendered Phase-2 **whole-scene** RGB-D, selected shared rigid pose/twist, per-frame objective/innovation, and Kabsch-vs-rigid trajectory.
7. Both candidate PRISMATIC/REVOLUTE joint image renders and axes **even if neither model is accepted**. Show joint-vs-free-rigid XYZ/pose residuals and image residuals.
8. Forward/backward toggle, actual chronological frame IDs, timestamps, camera names and fixed world transform. Full point buffers, no silently hidden point cap.

Browser verification must check actual rendered-image arrays versus numerical outputs, geometry projection into real camera pixels, predicted/observed/residual triplets, correct mask and depth units, and both candidates in unresolved runs. 'Viewer opens' is not acceptance.

## 9. Implementation plan and acceptance tests (must actually execute)

**Deliver changes, not another plan.** Work in this order and don't spend cycles refactoring unrelated pipelines:

**Gate A — Image renderer / operators.** Implement 1-camera and 8-camera colored point rasterization CPU and CUDA; image/color/depth/coverage oracle; finite-difference RGB and depth derivatives; `H/Hᵀ` adjoint identity; visibility and source-footprint corner cases; no background RGB/depth values read as valid measurements.

**Gate B — True image-based grouped IEKF.** Replace the production association-based observation path; verify signed-time process prediction, fixed prior `s⁻,P⁻` through iterations, correct grouped information equation, render/relinearize/true-posterior-energy line search, ONE posterior covariance calculation; all-camera participation, free per-point `[XYZ,v]`, correct supernode correction expansion, no group poses. On a textured rigid/translation/rotation synthetic RGB-D sequence generated with the same pixel geometry but **without providing point matches** to the estimator, demonstrate decreasing **observed image** RGB/depth errors and trajectory recovery. Verify dense small-scene covariance against the chosen correctly documented representation.

**Gate C — Coherence.** Pin reference 192-node graph, k=12, held-out node CV; correct moving side on static+moving toy object, no spurious motion for globally drifting/static scene; test from **image-estimated Phase-1 trajectories**, not only perfect synthetic trajectories.

**Gate D — Phase 2.** One temporal SE(3)+twist **Bayesian IEKF**, with the reference rigid prediction, prior information and Lie-group posterior covariance/reset; complete scene RGB-D renderer; no per-frame independent poses substituted for temporal motion; decreasing actual rendered-image objective and a physically coherent rigid trajectory on synthetic data.

**Gate E — Joint.** Exact `reduce_trajectory` diagnostics; pure prismatic -> PRISMATIC; finite-axis pure revolute -> REVOLUTE; static -> unresolved; insufficient excitation -> unresolved; errors reported in SE(3)/XYZ/image space. Validate actual image-space joint refinements and fixed-axis recovery.

**Gate F — Bidirectional/support.** Both endpoint catalogs, reversed processing with **chronological output**, correct `T(k) T(a)⁻¹`, all-camera image evidence and no double-counting; near-axis/occluded ambiguity, joint-guided expansion of known moving support on previously low-confidence measured endpoint points.

**Gate G — Real blade.** Fresh complete 60-frame x 8-camera forward/backward runs using the **image** measurement model (old all-camera association runs are explicitly incompatible caches). Record runtime and peak GPU memory on the 8-GB RTX 4070; framewise RGB/depth image losses, rendered/observed overlap, static reference evidence, core sizes, Kabsch unexplained motion, `gamma_1dof`, both candidate joint metrics, and forward/backward agreement. On success, run image-based joint-guided point classification. If real data remain unidentifiable, retain unresolved and supply a specific numerical/rendered-image failure—not an invented joint label.

**Gate H — Provenance/debug.** An executable CLI with input/output and both directions, all cameras default. Save versioned output manifest `measurement_model: rendered_rgbd_point_cloud_v1` and all renderer/IEKF source hashes; reject association-based caches. Preserve tests of old pointwise baseline. Show actual viewer screenshots and image residual CSV/JSON and the exact commands to reproduce them.

### Required final report

- **Implementation map:** source files/functions changed, explicitly state `associate()` is not in production Phase-1 / Phase-2 / joint observation paths.
- **Test table:** executed tests, numerical pass/fail, real scene status, no fabricated claims.
- **Per-frame/per-camera:** actual observed-predicted RGB RMSE, depth RMSE, coverage, residual heatmaps, IEKF update status, contributor counts, GPU/time.
- **Moving core:** representative-node count, stable support IDs, forward/backward counts and coherence evidence.
- **Rigid + joint:** Kabsch and temporal rigid trajectories, `gamma_1dof`, PRISMATIC/REVOLUTE pose and RGB-D image residuals, world axes, confidence and reasons for unresolved if any.
- **Completion:** static/moving/ambiguous point counts for each endpoint with corresponding image-evidence improvements.
- **Viewer:** verified working path and rendered-vs-observed screenshots.

## 10. Prohibitions and completion definition

**DO NOT:**

- Replace rendered-image residuals with nearest-neighbor XYZ, selected raw UVZ point factors, optical-flow matches, or reprojection-only residuals.
- Treat RGB only as data-association affinity. It is a **measured image loss** with valid spatial derivatives through the moving projected footprints.
- Use 3DGS Gaussian alpha, opacity training, black/white background compositing, hallucinated depth, or background-image supervision.
- Assume the old rank-24 per-group covariance factorization stays exactly valid when multiple groups touch the same rendered pixel.
- Reuse legacy association-based Phase-1 caches as if they came from the new measurement operator.
- Introduce physical supernode segmentation, camera split, arbitrary residual cutoffs, a new articulated-body state in Phase 1, or GT-based selection.
- Claim success only because SPD covariance, perfect supernode distance preservation, good point-association RMSE, or synthetic Jacobian tests pass.

**Success is the implemented and verified pipeline:**

`predict [XYZ,v],P⁻ → render colored point cloud into ALL cameras → grouped image-Jacobian IEKF with fixed prior and one posterior update → free point trajectories → RigEuler coherence → one temporal rigid SE(3) IEKF using the SAME image renderer → 1-DOF P/R reduction → forward/backward comparison → image-evidence static vs joint support completion`.

**The transported point cloud determines the nonlinear RGB-D measurement function. The full algorithm is a predicted-state/covariance Kalman recursion with an iterated render-based measurement correction and a once-per-frame posterior covariance update.**
