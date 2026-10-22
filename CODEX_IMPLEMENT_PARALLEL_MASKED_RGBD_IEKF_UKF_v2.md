# CODEX IMPLEMENTATION CONTRACT v2 — MASKED POINT REPROJECTION, TWO INDEPENDENT IEKF/UKF PIPELINES

**Authoritative specification. Read completely before editing code.**

Implement **two separate runnable pipelines** for the SAME point-based RGB-D SLAM outer loop:

- **Pipeline A — IEKF:** analytical, iterated error-state extended Kalman filter for rigid pose+twist, followed by Lie-group Rauch–Tung–Striebel (RTS) smoothing.
- **Pipeline B — UKF:** Lie-group sigma-point unscented Kalman filter for rigid pose+twist, followed by an **unscented** backward smoother using saved sigma cross-covariances.

**NO hybrid switching, NO UKF fallback inside IEKF, NO IEKF correction inside UKF.** Both share only data loading, compact point map, sparse projective association, the deterministic `SOLVED/UNSOLVED` loop, scoring, outputs, and CUDA kernels that do not implement either filter. Both use exactly the same dataset, calibration, initial map, sensor/process covariance assumptions, groups proposal logic, and acceptance objective.

The companion PDF `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.pdf` derives the state/output equations and CUDA mathematics. The theory PDF and its editable source `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md` are authoritative mathematical contracts; Section M1--M8 of that theory governs ALL foreground-mask, depth-validity, occlusion, and point-projection behavior and overrides older specifications. **Do not keep the previous eager per-pixel catalogue.**

## 0. Mandatory scope and prohibitions

**Goal.** The inner Kalman filter only predicts/corrects low-dimensional rigid transformations from corresponded RGB-D points. The SLAM outer loop maintains the currently `SOLVED` canonical points and `UNSOLVED[k]` raw observations. First-frame points initialize SOLVED. Every later frame is matched to the transported solved model, remaining observations retained in UNSOLVED. Perform a full forward filter and backward trajectory smoother, then discover late-visible points by reverse rigid transport and matching, then iterate chronologically through the temporal frontier to explain earlier UNSOLVED observations, and repeat the forward/backward/discovery cycle until no accepted improvement.

**Never implement:** 3DGS, Gaussian splatting, TSDF, surfels, surface normals, point-to-plane residuals, per-point independent velocity, per-pixel persistent variable, all-frames eager catalogue, per-pixel/per-point dense global covariance, global `3**N` enumeration, five fixed scene patterns, hardcoded static/dynamic labels, ground-truth motion/part labels for estimation, arbitrary XYZ novelty threshold, forced identity motion, silently assumed timing, or a dense pixels-by-pixels innovation matrix.

A static rigid group is the ordinary estimated solution `T[g,k] == I`. There is no separate static category. Every existing and new group uses the same state, matching, filter, and smoother. The only top-level semantic states are `SOLVED` persistent geometry and `UNSOLVED[k]` raw observation indices. Group IDs are *motion hypotheses*, not a third point category.

**Mathematical honesty.** Analytical IEKF Jacobians are derivatives of the *conditional matched-point output*, not an exact solution to unknown data association. UKF is a deterministic sigma-point approximation to the nonlinear measurement moments. Local association/model proposals are not exact global discrete inference. A monotonically accepted common score does not prove complete geometric recovery. Failure of fully hidden points to become identified is allowed. DO NOT claim a theorem proves more.

## 0A. CRITICAL: port the measured-data treatment, NOT the 3DGS renderer

These are **production requirements**, not optional enhancements; consult the updated PDF Section M1--M8 for complete mathematical derivations. Previous Codex stopping on an unset `visibility_rule` was an implementation error. No user-supplied visibility rule, confidence knob, hard-mask attraction gradient, or synthetic depth is permitted.

- Read the original 20 October 3DGS note's Eq. (41): `P=M*I`, `V_used=Vraw & M`; its RGB background Eq. (42) is only needed by the 3DGS **alpha renderer**, NOT by our point-only program. No `alpha`, `F`, `(A-M)^2`, random RGB background, or alpha JVP/VJP may appear in either production inner estimator.
- A point-based raw RGB-D observation exists **only** at `M=1 AND Vraw=1 AND finite(D) AND D>0`. Every new persistent point, matched target, unsolved candidate, and fusion source carries and validates that immutable pixel ID. RGB on mask-zero pixels and depth outside this set have ZERO role as measured points or motion residuals.
- For a solved point `p_i`, world point `x=T_gk p_i`, camera point `X=Rcw*x+tcw`, predicted pixel `pi(X)`, depth `X_z`, and fixed RGB `a_i`. A discrete matched valid target pixel `u_j` supplies the **single** conditional observation `z=(u_j.x,u_j.y,D_j,I_j)` versus `h=(pi(X),X_z,a_i)`. Pixel position + depth constrain motion; fixed target RGB mainly chooses association and has **zero pose derivative** once `u_j` is fixed. No normals/point-to-plane. No RGB compositing.
- The production IEKF must use exactly the predicted-output Jacobian from PDF Eq. (M7), derived from `Rcw*[I,-skew(x)]`. The production UKF must evaluate exactly the SAME matched raw-point output at each sigma state without a copied IEKF Jacobian; use the low-rank Woodbury updates in (M9)--(M10).
- The observed binary mask is an **association domain** plus conditional **hard pose-feasibility** for confirmed first-visible object points. Compute integer-bin z-buffer of transported SOLVED points (depth, pointID stable ordering); inspect valid foreground depth for ordering; preserve occluded SOLVED points without adding a false residual. Pixels outside image/with invalid foreground depth contribute no geometric target. A confirmed first-visible predicted foreground point landing at mask zero is infeasible **under the declared object-only/no-external-occluder data premise**; no background depth is permitted to rescue or oppose it. If external occluders are actually present and not modeled, explicitly mark that case unobservable rather than pretending mask-zero invalid depth proves empty space.
- Hard feasibility is checked on **trial poses**, outside `H` and sigma moment matrices. Do NOT differentiate a step-function z-buffer/mask condition. A rejected trial must roll back pose, covariance, associations and cache version. No preflight abort due to missing visibility parameters: projection rules are fully defined in PDF (M3).
- Preserve original foreground RGB and metric depth without smearing background RGB into samples. Optional direct photometric interpolation is OFF in baseline; only enable when interpolation support is valid foreground and `W R_raw W^T` with overlapping covariance is implemented/tested. Never combine its pixel RGB with the same pixel's frozen-correspondence RGB as if independent.
- For an observed point introduced at time `s`, transport it to time `k` by `T[g,k] @ inverse(T[g,s]) @ q_s`. Existing identity transforms need no special static pipeline. Unexplained measured foreground pixels remain indexed UNSOLVED.

**No unauthorized override:** Do not edit the 3DGS research renderer or static training; only the point-based RGBDekf pipeline. Keep the two filters completely independent and call the same shared projector, mask gate and solved/unsolved controller.

## 1. Audit the existing repository before replacing production code

Repository: `/home/samuelemara/Joint3DGS/RGBDekf` (or directory selected by user). Open its actual modules and produce `AUDIT.md` mapping each class/function to keep / rewrite / remove. Existing incorrect catalogue, fixed scene-pattern mixtures, observation-sized optimizer states, and full-catalogue per-image renderer must have **no callable path from either production `main.py`**. Preserve useful calibrated camera/dataloader logic if verified, tests and tiny reference code.

Produce independent runnable entry points:

```text
RGBDekf/
  main_iekf.py                      # independent analytical IEKF pipeline
  main_ukf.py                       # independent sigma-point UKF pipeline
  rgbd_common/
    config.py                      # resolved timing/camera/noise/prior config
    dataloader.py                  # read selected scene, stream all RGB-D frames
    geometry.py                    # ONLY canonical XYZ/color, ID, covariance handles
    lie.py                         # SO3/SE3 exp/log, adjoint, left/right Jacobians
    projection_cuda.py             # point transform, camera projection, visibility
    matching_cuda.py               # sparse local raw-pixel associations
    observation_model.py           # conditional z,h,R and unassigned likelihood
    fusion.py                      # canonical point fusion with provenance
    group_proposals.py             # deterministic correspondence-based rigid groups
    score.py                       # full comparable objective and local factor deltas
    outer_loop.py                  # THE SAME state machine for both pipelines
    outputs.py                     # all outputs, identical structure
    backends/
      torch_cuda.py                # Torch-CUDA fallback
      optional_extension.py        # optional fused CUDA C++ extensions
  iekf/
    dynamics.py                    # analytical f,F,G,Q(dt)
    measurement.py                 # analytical h,H and local Jacobian tests
    filter.py                      # inner IEKF, fixed prior per iteration
    smoother.py                    # analytical Lie RTS
    covariance.py                  # conditional and selected marginal queries
  ukf/
    dynamics.py                    # nonlinear sigma-point transition
    sigma_points.py               # manifold sigma-point generation/moments
    measurement.py                 # batch nonlinear h(sigma)
    filter.py                      # Woodbury UKF correction, NO dense innovation
    smoother.py                    # saved-sigma-cross-cov unscented smoothing
    covariance.py                  # UKF moment/cross-state consistency
  tests/
    test_camera.py
    test_lie.py
    test_data_association.py
    test_fusion_and_visibility.py
    test_iekf_math.py
    test_ukf_math.py
    test_covariance.py
    test_smoothing.py
    test_outer_loop.py
    test_gpu_equivalence.py
    test_real_dataset_smoke.py
  run_iekf.sh
  run_ukf.sh
  README.md
  AUDIT.md
```

Common code MUST NOT import either `iekf` or `ukf` directly. Outer loop uses a `MotionEstimator` protocol with `predict`, `correct`, `smooth`, and `refine_reverse_hypothesis`; the implementation injected by the chosen main is either *only* IEKF or *only* UKF. `main_iekf.py` never imports from `ukf`; `main_ukf.py` never imports from `iekf`.

## 2. Shared data loading / configuration contract

`python main_iekf.py --input '.../fridge_10905' --output '.../results_iekf' --noise-config '...yaml'` and the analogous UKF command must work with scene folders with spaces in paths and archives where supported. No hardcoded object names, frame count, cameras or FPS. Programmatically inspect actual selected directory file names, RGB, depth_npy, foreground masks, per-camera K/T and job metadata. Support all frames and all cameras by default. Masks enforce foreground-only observations and object-only first-visible pose feasibility per PDF (M1)--(M5); they do not indicate moving parts.

Required resolved configuration:

```yaml
scene:
  input: /path/to/selected_scene
  time: auto                  # read actual timestamp or FPS metadata, seconds
  use_all_frames: true
  use_all_cameras: true
  require_foreground_mask: true
  depth_unit: m              # verify original depth scaling
noise:
  source: user_approved_assumptions  # or independent_calibration
  p0: <read approved 12x12 state covariance, per group>
  q: <read approved process noise/spectral density with dt discretization>
  rgb: <read approved raw pixel covariance model>
  depth: <read approved depth covariance model>
  pixel_localization: <read approved association localization model>
  unmatched: <proper normalized observation/outlier model and prior>
compute:
  device: cuda
  accumulator_dtype: float64
  input_dtype: float32
  batch_pixels: auto_memory_bound
  deterministic_reductions: true
```

Read authorized numerical `P0/Q/R` guesses from the earlier source document in the actual repo and log the exact file, section and values in `run_config_resolved.yaml`; **do not invent numerical noise values or demand independent sensor calibration**. If the earlier numeric config is missing, fail with explicit missing fields (not a fake calibrated default). Search the actual selected input for frame timestamps/FPS; the fridge case previously resolved to **25 Hz**, but do not impose this on other scenes. Record `dt[k]` and unit seconds, and validate strict monotonicity.

The raw observation index `(k,c,u_x,u_y)` is immutable; source RGB/depth/valid/mask remain indexed on disk. Cache only streamed frames and compact ownership bitsets/CSR. Nonreturn readings need their declared likelihood or are labeled invalid/uninformative; never treat no-return as unconditional free-space evidence.

## 3. Common geometry state and ownership

Canonical solved point attributes: immutable `point_id`, `p_xyz[3]`, `color_rgb[3]`, group ID or documented mixture hypothesis, compact measurement information/covariance references, source raw observation ID, CSR provenance observations. Use contiguous SoA and GPU float32 XYZ/color with float64 information accumulators.

`SOLVED` starts as fused **frame-0** multiview masked depth points only. Do not mark them as having known motions merely because they are in `SOLVED`: group affiliations are initially provisional and can split when subsequent temporal correspondence evidence supports distinct rigid transforms.

`UNSOLVED[k]` begins as all valid foreground raw observations from `k>0`, represented by source IDs/ownership flags, **not 3D state variables**. Every matching pass updates observation ownership; points that temporarily disappear behind the measured depth remain solved. Duplicate camera observations of the same physical location are fused via measured correspondence/uncertainty, not concatenated forever.

Each group `g` has canonical chart ref frame `s_g`; its fixed-reference pose is `T[g,s_g]=I`. For any source point `q_s` acquired at `s`:

\[
 p_g=T_{g,s}^{-1}q_s,\quad q_k^{pred}=T_{g,k}T_{g,s}^{-1}q_s.
\]

This is a single **bidirectional** rigid transport formula. Never fit unrelated reversed transformations or invert velocity signs by hand.

## 4. Shared conditional RGB-D output equation (NO normals)

For solved point `p_i` assigned to group `g`, camera `c`, frame `k`:

\[
 x_{ik}=T_{g,k}p_i,\;X_{ick}=T_{cw,c}x_{ik},\;
 h_{ick}=\begin{bmatrix}\pi_c(X_{ick})^T&X_{ick,z}&a_i^T\end{bmatrix}^T,
\]

\[
 z_{kcu}=\begin{bmatrix}u_x&u_y&D_{c,k}(u)&I_{c,k}(u)^T\end{bmatrix}^T,
\quad z_{kcu}=h_{ick}+v_{kcu},\;v_{kcu}\sim N(0,R_{kcu}).
\]

Conditional `z` is 6D. Pixel location is an associated measured raw coordinate, not an independent continuous image rendering. RGB part is constant w.r.t pose for a fixed matched target pixel and fixed point appearance; RGB still informs matching. Optional true photometric warped residual needs `nabla I` and correctly correlated raw interpolated pixel noise. Do not double-count color evidence in two residuals.

Sparse matching: project currently solved points into each camera with a depth buffer, local pixel/tile candidate bins, and raw depth/RGB matching against the same observation model. Evaluate local competing assignments and an explicit `unmatched` alternative with normalized likelihood. No arbitrary Euclidean novelty cutoff. One raw observation is owned by at most one accepted canonical point (unless an explicitly derived sampling model says otherwise). Candidate unmatched observations remain `UNSOLVED[k]`. Test visibility/free-space/invalid/outside-camera cases, using depth uncertainty. Partial point projection does not imply dense predicted RGB-D coverage.

For performance, project/match only currently solved points into the next frame, and only sample raw pixels participating in local association. Backward discovery operates on indexed **unmatched** pixels in bounded spatial/time batches. Preserve all observations on disk for repeated replay.

## 5. Common outer controller — exact required temporal order

Create an explicit state machine with diagnostic traces, not a single monolithic optimizer:

```python
observations = Dataset.open(input_scene, all_frames=True, all_cameras=True)
solved = fuse_frame_zero_multiview(observations.frame(0))
unsolved = RawObservationOwnership.from_dataset(observations, frame0_provenance=solved.sources)
groups = bootstrap_group_hypotheses_from_correspondences(solved, observations)

for outer in iteration_stream():
    old_score = score_full_sequence(solved, unsolved, groups, observations)

    # PASS 1: FULL forward tracking, point map stays compact
    trace = motion_filter.forward(
        solved, groups, observations,
        match_fn=sparse_raw_rgbd_match,
        update_unsolved=unsolved,
    )

    # PASS 2: backward trajectory smoothing, same poses/covariances
    groups = motion_filter.smooth(trace)

    # PASS 3: late to early discovery from existing raw UN-SOLVED IDs
    for s in range(observations.num_frames - 1, 0, -1):
        hypotheses = reverse_match_unsolved(
            unsolved[s], observations, groups,
            relative_transport=lambda g,k: groups.T[g,k] @ inv(groups.T[g,s]),
        )
        # Existing group or proposed new rigid group; identity is a normal case.
        proposals = motion_filter.refine_reverse_hypothesis(hypotheses)
        transactional_accept_and_fuse(proposals, solved, unsolved, groups,
                                      score_full_sequence)

    # PASS 4: chronological frontier, each frame s searches earlier 0..s-1
    for s in range(1, observations.num_frames):
        refresh_ownership_at_frame(s, solved, groups, unsolved, observations)
        proposals = reverse_match_unsolved(unsolved[s], observations,
                                           groups, target_frames=range(s-1, -1, -1))
        solutions = motion_filter.refine_reverse_hypothesis(proposals)
        transactional_accept_and_fuse(solutions, solved, unsolved, groups,
                                      score_full_sequence)
        refine_affected_group_trajectory(motion_filter, solved, groups, observations)

    # PASS 5: new forward run is mandatory if new map/group edits were accepted
    accepted_score = score_full_sequence(solved, unsolved, groups, observations)
    assert accepted_score <= old_score + numerical_evaluation_error_budget
    log_full_outer_iteration(...)
    if no_accepted_edit_and_inner_stationarity():
        break
```

`transactional_accept_and_fuse` must log proposals including rejected ones, rollback any edit that increases the identical properly normalized full score, and update source-pixel ownership exactly once. A source pixel used to initialize a candidate may not be independently counted again as a measurement prior. A candidate surface seen only once and not constrained by any other observation may remain unsolved. The final output must retain unresolved observations honestly; do not fake 100% explanation.

Temporal frontier means **frame 1, then 2, ... K**, not just final-to-first and stop. An accepted new point or group must influence a subsequent forward estimation sweep. Two independent reverse trajectories are forbidden.

## 6. Pipeline A — analytical IEKF state, process and measurement

Implement in `iekf/` only. For group `g`:

\[
 s_{g,k}=(T_{g,k}\in SE(3),\nu_{g,k}\in R^6),\quad
 \delta s=(\delta\zeta,\delta\nu)\in R^{12},
\]

with `Tplus=Exp(delta_zeta^)Tbar`, `nuplus=nubar+delta_nu`. `nu` is a spatial/world twist with **translation-first** tangent ordering `(rho,phi)`.

\[
 T_{k+1}^-=Exp(\Delta t_k\widehat\nu_k^+)T_k^+,\qquad \nu_{k+1}^-=\nu_k^+.
\]

Let `A=Exp(Delta t nu^)`, `Tpred=A Tbar`, `phi=Delta t nu`. The exact first-order analytic linearized transition under this chosen chart is

\[
 F_k=\begin{bmatrix}Ad_A&\Delta t J_l(\phi)\\0&I_6\end{bmatrix},\quad
 G_k=\begin{bmatrix}Ad_{Tpred}&0\\0&I_6\end{bmatrix},\quad
 P^-_{k+1}=F_kP_k^+F_k^T+G_kQ_kG_k^T.
\]

The stated `G` corresponds to the source formulation's **right-applied** pose process noise; test numerical perturbations, including nonzero translations/rotations. If noise defined in a different frame, transform `Q` consistently; never mix charts.

For point `p_i`, `x=R_g p_i+t_g`, camera point `X=R_c x+t_c`, the 3D pose Jacobian is

\[
 J_{3}=R_c\begin{bmatrix}I_3&-[x]_\times\end{bmatrix},\quad
 H_a=\begin{bmatrix}J_\pi(X)J_3&0\\e_3^TJ_3&0\\0_{3\times6}&0_{3\times6}\end{bmatrix}.
\]

With conditional matched observations `(z,R)` and fixed predicted prior `(s^-,P^-)`, relinearize only the conditional model:

\[
 ((P^-)^{-1}+H_\ell^TR^{-1}H_\ell)\Delta_\ell=H_\ell^TR^{-1}(z-h(s_\ell))-(P^-)^{-1}\delta_\ell.
\]

Accumulate sparse per-group `12x12` blocks (and cross-group blocks only if factor physically involves multiple groups), and solve batched with Cholesky/LDLT. Hold `P^-` fixed during iterated correction. Optionally damp steps, but compute posterior covariance from the undamped local information and transport charts. Validate full analytic `F,G,H` against finite differences. Do not use generic Adam or GPU autograd as the IEKF correction.

**Backward A:** Save `F[k]`, `P_filt[k]`, `P_pred[k+1]`, filtered/predicted states. Gain `C_k=P_filt[k]F_k^T(P_pred[k+1])^-1`; smooth pose and twist by `boxplus(C_k (s_smooth[k+1] boxminus s_pred[k+1]))`; covariance `P_smooth[k]=P_filt[k]+C_k(P_smooth[k+1]-P_pred[k+1])C_k^T`, transported to the new chart. This is the IEKF branch's RTS; do not copy the UKF cross-cov smoother into it.

## 7. Pipeline B — Lie-group UKF state, process and measurement

Implement in `ukf/` only, same `s=(T,nu)` and dimension `n=12` per group, same nonlinear process `f` and matched-point nonlinear output `h`. No IEKF analytical measurement Jacobian used to update UKF states.

**Prediction.** Joint sigma points over state error `n=12` and process noise `q=12` (49 for one group). `P_aug=blockdiag(P_filt,Q)` for independent state and *new* process noise; if correlated, use actual joint covariance. Sigma coefficients `alpha,beta,kappa` resolved in config and logged, with `lambda=alpha^2(n_aug+kappa)-n_aug` and `n_aug+lambda>0`. Positive-weight option `(alpha,beta,kappa)=(1,2,0)` yields `lambda=0`, positive covariance weights and numerical stability. Generate ± columns of `chol((n_aug+lambda)P_aug)` around zero, retract onto `SE(3)`, run nonlinear dynamics for EACH sigma, and reconstruct a consistent Lie-group mean/covariance; store predicted--previous cross covariance from the *same* augmented sigma ensemble. Do not use EKF `F` to fill this cross covariance.

**Measurement.** For fixed externally matched raw observation factors, generate new `L=2n+1=25` sigma states from `(s_pred,P_pred)`, compute *nonlinear* `h_a(sigma_j)` for every active match in GPU chunks. Keep only `LxL` aggregated covariance information, never `M_raw x M_raw` innovation matrix. With nonnegative covariance weights define `A[:,j]=sqrt(wc_j)*(sigma_state_j boxminus mean_state)` and `Z_a[:,j]=sqrt(wc_j)*(h_a(sigma_j)-hbar_a)` and `e_a=z_a-hbar_a`. Accumulate

\[
 M=I_L+\sum_a Z_a^T R_a^{-1}Z_a,\quad
 b=\sum_a Z_a^T R_a^{-1}e_a.
\]

For correlated reused raw pixels use shared sparse `R^{-1}` operator rather than independent sums. Woodbury update:

\[
 \delta s=A M^{-1}b,\qquad P^+=P^- - A(I_L-M^{-1})A^T.
\]

This is algebraically exact **for the UKF sigma-derived moments**; it is not globally exact nonlinear inference. The same conditional observation vector is used as in IEKF. Evaluate accepted state against the conditional score; a UKF step need not be an objective descent direction. Do not import IEKF to rescue rejected steps. Instead retain previous state, revise match hypotheses, or perform another separate UKF sweep.

**Backward B:** save sigma-based `P_cross[k,k+1]`; `G_k=P_cross[k,k+1] (P_pred[k+1])^-1`, then unscented smoother mean update and covariance update using that `G`. Correct Lie chart transport. No analytical `F` substitution or independent reverse trajectory averaging.

**Expected GPU:** `[25,N_active,3]` sigma-transformed point chunks and `25x25` reduction for each independent group; process prediction uses 49 sigma points. Coupled groups require joint sigma dimensions and corresponding small covariance blocks. Do not invent separability when pose variables share an association or map factor.

## 7A. Required CUDA-facing APIs (identical common code for both estimators)

Build and test these **production** interfaces (names may adapt existing repo names but the contracts must hold):

```python
# native bool mask, explicit raw depth validity. Never use depth>0 alone.
fg_ids = compact_valid_fg(mask_hw, valid_depth_hw, depth_hw)
xyz, rgb, raw_ids = backproject_fg(fg_ids, depth_hw, rgb_hwc, Kinv, Twc)
uv, z, pixbin, in_view = transform_project(points_xyz, group_ids, poses, K, Tcw)
first_index, first_depth = zbuffer_first(uv, z, pixbin, in_view, point_ids)
edges_csr, owned_raw, unsolved_raw = match_masked_raw_points(
    predicted_uv=uv, predicted_depth=z, fg_ids=fg_ids,
    rgb=rgb_hwc, depth=depth_hw, valid=valid_depth_hw, mask=mask_hw,
    zbuf=first_depth, noise_config=declared_noise, ownership='unique')
pose_is_feasible, violations, unknown = mask_feasibility_trial(
    uv, z, first_index, first_depth, mask_hw, depth_hw, valid_depth_hw,
    object_only_premise=True)
# A only: H exact analytic, RGB pose rows zero for fixed association.
A12, b12 = iekf_segmented_information(edges_csr, fixed_raw_targets, poses, points, covariance)
# B only: reproject same associated raw IDs under each sigma pose.
G25, q25, As = ukf_sigma_moments_woodbury(
    edges_csr, fixed_raw_targets, sigma_poses, points, covariance)
# Shared reverse projection, no independent reverse geometry renderer.
reverse_xyz = transport_relative(q_at_s, T_g_at_k, T_g_at_s)
```

The API is illustrative of tensor semantics, **not** permission to replace the theory's proper association/noise model with simple Euclidean `dist<threshold`. All GPU correspondence scoring must read *only indexed valid raw foreground pixels*; the mask and color/depth are immutable, and all active factors reference raw IDs.

**Memory:** compact `N_solved x 3` point arrays, one bounded decoded frame, sparse CSR `E` edges, `H x W` z-buffer, small `12 x 12` IEKF group information or `25 x 25` UKF Woodbury matrices; no cataloguing every point across all frames, no `N_point x N_pixel`, no `N_pixel x N_pixel`, and no sigma copies of all frames. `float32` projection, `float64` small accumulation/Cholesky. If multi-group posterior couples groups, keep sparse/block factors; do not pretend per-group independent covariance is exact.

`first_index` is a deterministic **self-occlusion** check, NOT permission to classify any masked-out depth sample as an observed occluder. For actual foreground occlusion, consult only `M=1 && valid` depth. Any object-only hard-mask failure is evaluated on the nominal **trial**; sigma points crossing different discrete association or visibility branches are not silently assigned a fictitious continuous measurement.

GPU kernel tests must compare against a small pure Torch/NumPy reference on randomized nonzero poses, multi-camera optical axes, pixel boundaries, identical depths, behind-camera points, mask 0 and depth invalidity. Cache entries are keyed/invalidate on pose, map, ownership, camera, mask/version and candidate state.

## 8. GPU kernel contracts and bounded memory

Implement a pure PyTorch CUDA path and optional fused C++/CUDA kernels. Required GPU tasks:

1. `transport_and_project(points, group_ids, poses, cameras) -> (xy,depth,valid,point_ids)` in SoA, one active map point per CUDA thread; output per camera on demand.
2. `build_pixel_bins_and_zbuffer(...) -> CSR offsets, candidate ids, visible ids` via deterministic sorting/segmented minimum and explicit tie policy; retain predicted image validity mask.
3. `match_rgbd(raw_frame, projected, priors) -> CSR matches, raw ownership, unmatched ids, local likelihood factors` using sparse local search; unassigned option in identical score; camera-space residuals against **raw** depth/XYZ/RGB.
4. `iekf_reduce(matches) -> A[B,12,12],b[B,12]` float64 segmented reduction; analytic `H`, never materialize global dense Jacobian.
5. `ukf_project_sigma(chunks) -> h[L,chunk,dim]`; `ukf_reduce_moments -> M[L,L],b[L]` float64 accumulation; no global innovation covariance.
6. `inverse_transport_unmatched(raw_points, source_frame, target_frames, group_poses)` uses `T_target @ inverse(T_source)` and shared projector/matcher for backward discovery.
7. `fuse_canonical_points(corresponded_points, group, provenance)` performs small linear/information updates and records pose--point correlations, not independent duplicate XYZ for each camera.

**Memory invariant:** persistent device memory `O(N_solved + max_frame_raw + K*(12B)^2 + sparse_map_factors)`. Temporary `O(N_active*L_chunk + |E|)` for UKF and `O(N_active + |E|)` IEKF, with explicit chunking. No `O(N_total_observations*B)` persistent catalogue and no dense `N_map^2` or `N_pixels^2` structures. Monitor `torch.cuda.max_memory_allocated` and RSS separately. Assert GPU peak below installed card usable memory on real run.

**Accuracy:** float32 or optionally float64 points, float64 small information/moment accumulators; compare GPU to float64 CPU reference at random camera poses, including large rotations, negative Z, masks, overlapping pixels and occlusion. For fused CUDA code, provide a reference Torch-CUDA backend and automated parity tests.

## 9. Covariance, map fusion, and measurement independence

For a later observed world point `y_s` inserted under group pose `T_s=(R_s,t_s)`, canonical `p=T_s^-1 y_s`. Under the **left pose perturbation**,

\[
 J_y=R_s^T,\quad
 J_T=-R_s^T[I,-[y_s]_\times],\quad
 P_{pp}=J_yR_yJ_y^T+J_TP_{TT}J_T^T,
\]

when source measurement and pose errors are initially independent; include cross terms if not. Importantly, `P_{pT}=J_TP_{TT}`, and two points born under a common pose have `P_{p_i p_j}=J_{T,i}P_{TT}J_{T,j}^T` plus shared measurement terms. Represent via sparse source-pose factors or selected covariance queries, NOT a dense point-point matrix. If the inner filter conditions on a fixed solved map, label reported `P` **conditional**. Marginal requires Schur/factor-graph selected queries. Do not count one raw source observation as a landmark prior and again as independent target measurement.

Pixel association `(u,v)` may be uncertain; use a declared pixel-localization noise model. If using bilinear RGB sampling for direct image alignment, propagate full covariance of repeated raw pixel reuse, including off-diagonal shared weights. Raw measured depth and the backprojected XYZ are the same source of information; don't count as independent.

A newly discovered point may not provide any pose information if observed only at its source time: source-frame self-fit is noninformative. Cross-frame matches and backward visibility supply actual evidence. Unidentified hypotheses remain in `UNSOLVED`.

## 10. Full score and admissible convergence claims

Define a **single comparable full-sequence score**:

\[
 J = \sum_{\text{owned raw pixels}} -\log p(z\mid p,T,g,a)
   +\sum_{\text{unassigned raw pixels}} -\log p_U(z)
   +J_{motion-prior}+J_{geometry-prior}+J_{group/association-prior}.
\]

Each accepted outer change must decrease or leave unchanged **this exact same evaluation measure with identical observation domain, data provenance and normalization**. Reject and rollback changes otherwise. Inner IEKF conditional objective may decrease while outer global score does not; report both. UKF's moment update is not guaranteed to descend a MAP objective, so measure and reject non-improving outer proposals, without switching filters. A bounded-below accepted score sequence converges *in value*. Do not claim global optimality, exact full Bayesian posterior, or a guarantee that all observed points can be solved.

When candidate count varies, use proper priors/normalization so old-vs-new objectives are actually comparable. If model evidence cannot be reliably evaluated, report an unresolved mathematical limitation instead of adding a numerical threshold.

## 11. Mandatory unit, integration and real-data tests

### 11A. Mathematical correctness

- Lie `Exp/Log`, `Ad`, adjoint rotation/translation ordering and group composition/inversion.
- IEKF `F,G,H` finite differences in left Lie chart (random nonzero pose/velocity, actual camera convention).
- IEKF correction vs exact linear Gaussian Kalman update on an equivalent linear fixture; repeated inner relinearizations keep fixed predicted prior.
- Joseph-form conditional covariance reference; posterior covariance unaffected by optimization damping.
- Analytical RTS state and covariance against small dense batch Gaussian smoother.
- UKF sigma points reproduce Gaussian mean and covariance; augmented nonlinear prediction deterministic and consistent.
- UKF matches exact linear Gaussian update for linear measurement; `25x25` Woodbury vs explicit small dense innovation `S` for a tiny measurement fixture.
- Unscented backward gain from saved sigma cross-covariance; compare with RTS under a linear process fixture.
- Covariance augmentation for late points: correct point-pose and shared-pose point-point cross-covariance; test with Monte Carlo only as an external numerical verification, not as inference.
- Shared raw interpolated noise and masks; no duplicated independent measurement evidence.

### 11B. Correspondence and geometry

- Multi-camera synthetic point appearing twice fuses to one persistent point with two provenance IDs.
- A point observed at `s` inverse-transports to earlier `k` under `T_k T_s^-1`, and forward returns to its source position.
- Occluded predicted points receive no RGB-D correction. Under the declared object-only premise, confirmed first-visible point projections at mask zero make candidate poses infeasible, never a synthetic depth residual. Missing depth/off-camera supplies no fabricated evidence.
- One initially visible object splits into two rigid groups only from genuine interframe correspondence evidence; identity trajectory is permitted without a static flag.
- Late-visible new points first remain UNSOLVED, then join SOLVED following actual backward multiframe matches; source observation counted once; fused point modifies next forward filter.
- No feature depends on shape normals, surface meshing, TSDF or Gaussian functions.

### 11C. Full pipeline mandatory acceptance

- Loader reads **every time and every camera** for the selected scene, camera transforms verified on overlapping RGB-D measurements; metadata FPS/timestamps logged.
- Each pipeline independently completes **forward -> own smoother -> backward discovery -> temporal frontier -> second forward**, recording accepted and rejected edits, and real trajectory updates.
- Report total solved point count at frame 0, after reverse sweep and final; per-frame unmatched raw observation count; group count; trajectories/twists and conditional/marginal covariance labeling; RGB-D matching score and full objective.
- Export `solved_points.ply`, `solved_points.csv`, `unsolved_observations.csv`, `group_poses.npz`, `group_twists.npz`, `group_covariance.npz`, `source_provenance.npz`, `objective_history.csv`, `stage_log.csv`, `run_config_resolved.yaml`, `validation_report.json`, and projected point RGB/depth/valid masks for chosen frames in both temporal directions.
- Run the **same** scene once with IEKF and once with UKF; use identical initialization/data/matching/noise models, while recording runtimes and peak GPU memory separately. No results from a two-frame smoke test may be presented as full-scene success.
- Log success/failure per acceptance gate and attach exact command lines, without claiming that filtering automatically discovers an arbitrary number of perfectly segmented bodies.

## 12. Execution and development sequence (mandatory order)

1. Read this file and the companion PDF fully; inspect repo and selected actual input dataset; produce audit and resolved metadata/noise contract.
2. Implement and test common dataloader, Lie group, raw point backprojection, frame-0 fusion, sparse GPU transport/projection/matching and ownership tracking. Show memory allocation per module.
3. Implement IEKF `f,F,G,Q,h,H`, filtering/covariance, own RTS smoother; mathematical unit tests.
4. Implement UKF nonlinear process sigma prediction, low-rank Woodbury measurement, own unscented smoother; independent unit tests. Do not alias to IEKF.
5. Implement shared outer state machine and proposals/transactional score, including late-to-early and chronological frontier steps.
6. Run minimal deterministic synthetic tests for both, compare all math to reference solutions; then real-data smoke tests without reporting reconstruction success prematurely.
7. Run both complete pipelines on the full selected scene; include actual second forward pass after geometric discovery, save structured results and validation.
8. Audit constraints and document any real mathematical ambiguity or computational failure. Fix architectural failures before performance-only tuning. **Do not turn this into another catalogue optimization.**

## 13. Commands to deliver

Provide both examples with the actual validated main flags and paths:

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
python main_iekf.py --input '/path/to/selected_scene' --output '/path/on/Extreme SSD/results_iekf' --noise-config './config/approved_noise.yaml'
python main_ukf.py --input '/path/to/selected_scene' --output '/path/on/Extreme SSD/results_ukf' --noise-config './config/approved_noise.yaml'
pytest -q
```

Do not claim flags exist before implementing them. Use dedicated SSD output, logs and temporary directories, with safe shell quoting; do not modify the input dataset or ground-truth files. No unnecessary install/new CUDA package should be required beyond the named `rigeuler3dgs` environment; custom extension builds require documented prerequisites and GPU fallback.

## 14. Required final Codex report

Report independent results **A: IEKF** and **B: UKF** in the same metric table. Separate: loaded frames/cameras, initial/final solved points, unexplained raw observations, number of groups, forward passes, reverse discovery passes, frontier passes, accepted objective history, pose/covariance sanity checks, CUDA peak memory, runtime, tests passed/failed. Show actual paths to saved data. List all incomplete criteria honestly. Do not substitute a successful initial likelihood computation, smoke test, visualization or component test for a completed full algorithm.


## 15. UPDATED foreground-mask and point-output acceptance matrix (all mandatory)

**Run every test separately on IEKF and UKF except filter-specific derivative/moment mathematics.** Save `MASK_PORT_VALIDATION.json` recording actual assertions, test names, branch, GPU, and raw source IDs.

1. Check pixelwise preparation `P=M*I`, `Vused=Vraw & M` against 3DGS Eq. (41) as a *source consistency audit*; point renderer uses only `Vused` observations and does **not** composite RGB over artificial backgrounds.
2. Actual dataset uses correct camera/calibration/timing; accepts 0/1 or 0/255 mask, rejects other unlabeled mask intensities; scans all 328x3 if that is the selected real scene (never hardcode those counts).
3. Replace every mask-zero RGB with arbitrary random RGB; identical forward and backward assignments, IEKF information, UKF moments, discovered point IDs and full point score.
4. Replace every invalid or mask-zero depth value with NaN, zero, extreme positive depth, random; identical point targets, assignments, corrections and insertion.
5. Every frame-0 source pixel, later matched target, UNSOLVED candidate, and fused persistent point passes `mask && Vraw && finite_positive_depth`. Zero leakage in production CSR and source provenance.
6. For a valid FG target, conditional output `z/h` and fixed-RGB pose zero row tested on CPU/GPU; IEKF analytic $(uv,d)$ derivative matches finite differences, including non-identity camera and pose.
7. UKF 25-sigma-point moments and streaming Woodbury step reproduce explicitly formed small `S` reference; no IEKF call path; augmented sigma predictor/unscented smoother retain own cross-covariance.
8. Model-first z-buffer frontmost tie-break stable; valid nearer foreground target occludes solved point without deleting it or creating false factor. Out-of-frame/invalid FG depth creates no target.
9. Under object-only premise, a confirmed first-visible model point at negative mask yields infeasible candidate **with rollback**; no need for `visibility_rule` or `confidence` config, and no synthetic negative-depth residual.
10. Test late point discovered in frame `s` transported backward under `T[k]*inv(T[s])`, naturally hidden in frame zero; source observation counted once; map expands and changes the next actual forward pass.
11. Unique ownership accounting per (frame,camera), `valid_foreground = owned + unsolved`; no raw pixel independent double count or all-frame persistent point hypotheses.
12. Overlapping bilinear samples generate expected nonzero cross-covariance if optional photometric mode is ON; if not implemented, assert photo mode disabled and RGB influences only matching.
13. Measure peak GPU memory / CPU RSS and benchmark sparse projection, matching, IEKF reductions and UKF 25x25 update; no quadratic memory scaling in total point observations.
14. Both real pipelines complete **first forward filter > own backward smoother > actual reverse discovery > progressive temporal frontier > second complete forward filter**, and save objectively verifiable stage trajectories, point IDs and model score histories. Test results cannot be replaced by a completed preflight, a first objective or numerical unit tests.
15. No 3DGS alpha code or Gaussian renderer is reachable from the two production mains. No normals, surfels, explicit static/dynamic labels, hidden visibility policy, or background RGB/depth as measurements.

If the GPU algorithm or association model remains incomplete, report **FAIL** for that exact criterion and the line/path of the remaining blocker. Do not claim `100% passing unit tests` implies full-scene reconstruction. Include a table connecting PDF equations (M1)--(M11), IEKF (A1)--(A8), UKF (B1)--(B7), and CUDA (D1)--(D10) to tested source functions.

## 16. Source, precedence and authoritative execution prompt

Read this file, then the **entire** `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.pdf` and its Markdown source. The user-supplied `Bidirectional_Articulated_3DGS_IEKF_Masked_RGB_Valid_Depth_2026-10-20(1).pdf` is the source **only** for its RGB mask and depth validity, NOT for transferring Gaussian alpha into point reconstructions. PDF M1--M8 prevails over any earlier contradictory hard-mask or visibility rule.

Implement now in the actual RGBDekf repository, with both completely independent production filters and one shared solved/unsolved controller. Validate on real full input, keep records on the SSD, and state failures honestly. **Do not stop at planning, preflight, or synthetic/unit-test success.**
