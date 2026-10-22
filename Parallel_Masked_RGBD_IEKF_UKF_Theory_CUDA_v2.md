---
title: "Two Independent Sparse IEKF and UKF Frameworks for Masked RGB-D Point SLAM"
subtitle: "Exact foreground observation preparation, point reprojection, GPU sparsity, and independent forward/backward filtering"
author: "Mathematical and CUDA Implementation Specification"
date: "20 October 2026 (revised masked point-reprojection edition)"
lang: en
abstract: |
  This document specifies two independent point-only RGB-D SLAM algorithms.
  Both share a solved/unsolved outer loop, binary-mask-constrained direct multiview point reprojection,
  local association, reverse discovery of late-visible geometry, and one
  canonical rigidly transported point map. Framework A uses an analytical
  iterated error-state extended Kalman filter and an RTS smoother. Framework B
  uses a Lie-group unscented Kalman filter, a low-rank Woodbury measurement update,
  and an unscented smoother. The mask/depth semantics are ported from the supplied 20 October 3DGS note, but its Gaussian alpha/random-background terms are not. Neither a hybrid filter nor a Gaussian/surface
  renderer is permitted. Convergence claims are explicitly limited to what the
  specified algorithms and acceptance conditions support.
---

# Executive model and method boundary

The **only persistent geometry** is a compact set of canonical points whose group motion is currently explained (`SOLVED`). The remaining multiview RGB-D pixel observations at each timestamp are **indexed data** (`UNSOLVED`), not unknown point-state vectors. Frames are replayable. Groups share rigid $SE(3)$ transformations, including possible identity trajectories; *static* is not a separate category.

The **outer SLAM loop** is strictly common to A and B: fuse frame-0 observations; forward match solved points and estimate rigid motions; backward smooth those same motions; use late-unexplained observations to discover earlier-occluded points and group hypotheses; revisit earlier frames in chronological frontier order; repeat under one comparable score. The **inner filter** only refines low-dimensional rigid pose+twist using already-proposed correspondences and the chosen noise model.

This document does **not** claim that either Kalman approximation can provably recover all physical surfaces, solve unrestricted global correspondence exactly, or infer the motion of geometry that was never adequately observed. Its equations specify the requested estimators and the testable conditions for accepted changes.

| Property | Framework A | Framework B |
|:--|:--|:--|
| Motion state per rigid group | $T\in SE(3)$, spatial twist $\nu\in\mathbb R^6$ (12D error) | The same state (12D error) |
| Predict | Analytical $f,F,G,Q$ | Nonlinear augmented sigma points |
| Correct | Sparse analytical $H$ and information IEKF | Nonlinear sigma-output moments and Woodbury solve |
| Backward smoother | Lie-group RTS, saved $F$ | Unscented RTS, saved sigma cross-covariance |
| Filter Jacobian | Required for measurement update | Not used for measurement update |
| Outer SLAM map and association | Shared byte-identical code | Shared byte-identical code |
| Point geometry | XYZ + RGB, no normals/surfaces | XYZ + RGB, no normals/surfaces |
| Statistical statement | Local Gaussian approximation and local descent with damping assumptions | Deterministic sigma approximation; no generic MAP-descent theorem |

# Reading guide

The common solved/unsolved contract is normative for both algorithms. The next two chapters derive **complete and separate** state-space and smoothing formulations. The final chapter gives the GPU, covariance, and performance implementation contract and a rigorous testing plan. The accompanying Codex Markdown maps these formulas to exact module boundaries and runnable acceptance tests.

# Masked RGB-D point-reprojection revision (20 October 2026)

**Normative amendment, shared by A (IEKF) and B (UKF).** This chapter replaces any older text implying alpha compositing, photometric background carving, metric background depth, a configurable visibility classifier, a point-cloud surface normal, a differentiable mask-distance penalty, or a continuous image renderer. Both filters retain **the same SOLVED/UNSOLVED outer loop**, and neither calls or switches to the other filter. The 15-page supplied 3DGS source is used here for its *observation preparation* and information-form discipline, not for its Gaussian-renderer-specific output map.

## M1. Exact correspondence to the 3DGS source and the output-model boundary

The supplied `Bidirectional_Articulated_3DGS_IEKF_Masked_RGB_Valid_Depth_2026-10-20(1).pdf`, Section 13, declares the original measured RGB $I_{kcu}$, metric depth $D_{kcu}$, independent raw depth validity $V^{raw}_{kcu}$, and native binary whole-object mask $M_{kcu}\in\{0,1\}$. In its Gaussian output model it prepares

$$
P_{kcu}=M_{kcu}I_{kcu},\qquad V^{D,used}_{kcu}=V^{raw}_{kcu}\wedge(M_{kcu}=1),\qquad Y^C_{kcu}(b)=P_{kcu}+(1-M_{kcu})b.
$$

Its *renderer* instead predicts $C=F+(1-A)b$ where $F$ is premultiplied Gaussian RGB and $A$ is alpha. Therefore it derives an exact quadratic expectation with an $(A-M)^2$ term under artificial random RGB backgrounds. **That last identity has no direct point-reprojection equivalent**, because a point does not have Gaussian opacity or a continuous alpha footprint. Do not port random artificial backgrounds, $A$, $F$, Gaussian alpha JVP/VJP, or the $(A-M)^2$ penalty. Do not interpret the source's static transparency-carving experiment as validation of foreground point motion. Replacing $A$ with an unmodeled binary occupancy mask would produce discontinuous or undefined pose gradients, not a Gaussian-Jacobian substitute.

The *transferable* contract is instead the exact restriction to raw foreground measurements and the elimination of all metric depth information outside the real valid masked depth pixels:

$$
\boxed{\Omega^D_{kc}=\{u:\ M_{kcu}=1,\ V^{raw}_{kcu}=1,\ 0<D_{kcu}<\infty\}.}
\qquad\text{(M1)}
$$

Let $\Omega^M_{kc}=\{u:M_{kcu}=1\}$ for optional binary mask-only visibility checks. $\Omega^M$ does not define a 3D observed point without valid depth. No background RGB, background depth, invalid foreground depth, or 3D synthetic ``far plane'' is a point correspondence, geometric residual, or new canonical point. Missing depth is **missing**, not measured empty space. The original depth-valid bit must be read independently of the mask, then intersected with it. Depth stored as zero outside validity never enters a loss or covariance matrix.

If raw masks are 0/255, normalize by *exact* discrete equality, not by an arbitrary threshold. Prefer native-resolution RGB/depth/mask so the pixel grid is shared. If resizing is unavoidable, specify camera-intrinsic and depth resampling transforms; a fractional resampled RGB coverage mask must **not** silently become binary depth validity. Background RGB changes with all $M=0$ must have no effect on either filter, association, or promoted geometry. Background depth changes must likewise have no effect because it is invalid as an observation.

## M2. Sparse output map and disjoint observation ownership

The persistent canonical map consists solely of tuples $(p_i,a_i,g_i,\mathcal P_i,\mathcal C_i)$: XYZ, RGB, rigid-group reference, immutable source-pixel provenance, and compact geometric uncertainty/correlation references. It contains no normals, surfels, mesh patches, Gaussian opacity, or a dense image representation. A group with $T_{gk}=I$ is an ordinary solved rigid group, not a special static/dynamic category. Raw observations remain keyed by $(k,c,u)$, and at each frame the two conceptual sets are

$$
\mathrm{SOLVED}=\{i:\text{persistent point with supported rigid motion}\},\qquad
\mathrm{UNSOLVED}_k=\Omega^D_k\setminus\mathrm{OWNED}_k .
\qquad\text{(M2)}
$$

$\mathrm{OWNED}_k$ is a set of raw observed pixel IDs assigned to solved geometry, not a duplicate copy of every frame's point cloud. For each $(k,c)$, exact bookkeeping is $|\Omega^D_{kc}|=|\mathrm{OWNED}_{kc}|+|\mathrm{UNSOLVED}_{kc}|$ if ownership is unique. If an observation has multiple predicted candidates, settle a single primary ownership or explicitly retain its correlated/multi-hypothesis statistical model; never add independent equal-weight factors reusing one depth/RGB pixel.

For solved $i$ in group $g$, under $T_{gk}=\begin{bmatrix}R_g&t_g\\0&1\end{bmatrix}$ and camera $T_{cw,c}$, let

$$
 x_{ik}=R_{gk}p_i+t_{gk},\quad X_{ikc}=R_{cw,c}x_{ik}+t_{cw,c},\quad \hat u_{ikc}=\pi_c(X_{ikc}),\quad \hat d_{ikc}=e_3^\top X_{ikc}.
\qquad\text{(M3)}
$$

For a fixed raw-pixel association $a=(i,k,c,u_j)$ with $u_j\in\Omega^D_{kc}$, define **one** conditional observation and prediction

$$
 z_a=\begin{bmatrix}u_j\\D_{kc}(u_j)\\I_{kc}(u_j)\end{bmatrix},\quad
 h_a=\begin{bmatrix}\hat u_{ikc}\\\hat d_{ikc}\\a_i\end{bmatrix},\quad r_a=z_a-h_a,\quad r_a\sim\mathcal N(0,R_a)\text{ conditionally on association}.
\qquad\text{(M4)}
$$

Pixel-coordinate uncertainty represents quantization, calibration, localization, and association modeling, not independent noisy ``sensor pixel positions'' automatically. In this fixed-correspondence model the RGB residual contributes to *matching and point color estimation*, but has **zero pose derivative** if $a_i$ is held fixed. It is invalid to claim that $(I(u_j)-a_i)$ alone moves the pose after fixing $u_j$. Optional warp-based color alignment $I_{kc}(\pi_c(X))-a_i$ may give nonzero gradient, but only with all interpolation source pixels foreground-valid, and correctly propagated raw-pixel correlations $R_{sample}=W R_{raw} W^\top$; otherwise keep that option OFF. Do not stack warped RGB and the same raw RGB value as independent factors.

A pixel/metric-depth residual already describes a 3D measured point under camera projection. Do not add an independent Euclidean XYZ residual derived from the same depth pixel without carrying the exact induced cross-covariance. No *dense* photometric image reconstruction score is implied where projected points do not cover pixels: unclaimed valid foreground measurements belong to UNSOLVED.

## M3. A deterministic, executable projection/occlusion procedure: no visibility policy guard

For each candidate pose and camera use the following **fixed algorithm**, common to both filters.

1. Project solved points with $X_z>0$ and in-image coordinates. Pixel bin $q(\hat u)$ is the integer pixel center/bin under the declared camera raster convention. No continuous splat kernel is introduced.
2. In each pixel bin form the frontmost predicted solved point using $\arg\min_i(\hat d_i,\mathrm{pointID}_i)$; ties use deterministic ID order. Store `first_index[H,W]` and `first_depth[H,W]`, not an $N\times H\times W$ tensor. This is **model self-occlusion ordering**, not evidence from background depth.
3. At $q$, consult $M(q)$ first. If $M(q)=1$ and real depth is valid, compare its measured foreground depth $d_{obs}$ with $\hat d$ and use a foreground match or recognized foreground occlusion. Depth agreement is evaluated through the same declared noise distribution and association cost (not by an invented $|d-d_{obs}|<\epsilon$). If a closer observed foreground depth explains why a model point is unseen, **retain the point** and create no false missing-point residual.
4. If $M(q)=0$, no raw RGB-D point measurement exists at $q$. A candidate **confirmed first-visible object point** falling in this negative support contradicts the object-only mask interpretation. Do not use RGB/depth at $q$ to pull it back; reject that trial or mark the hypothesis infeasible. Crucially, because masked-out *background* depth is invalid, unmodeled external occluders cannot be excluded from RGB-D data alone. Thus first-visible is confirmed only conditional on the dataset's **object-only/no-external-occluder premise** and the known projected map. Where that premise is false or unknown, the point's visibility status is **undetermined** rather than an artificial impossibility. This limitation must be reported, not solved by a new user-configured confidence threshold.
5. If $M(q)=1$ but metric depth invalid, there is no 3D measured target. Its mask alone does not create an RGB-D residual or a new point. If $q$ lies outside the calibrated image, this camera imposes no mask constraint.
6. If the predicted point is behind another projected solved point or a valid closer foreground observation, its observation is occluded; it remains in SOLVED and receives **no motion correction from that ray**. Pixels in $\Omega^D$ unclaimed after assignment remain UNSOLVED.

The depth comparison can be made with a **noise-derived likelihood** $p(d_{obs}\mid \hat d,\sigma_D)$ and the declared unassigned/occlusion likelihood. A comparison sign $d_{obs}<\hat d$ states geometric ordering, whereas accepting a match or distinguishing noise and occlusion is a model-selection decision. Never hard-code an arbitrary depth tolerance, a free ``visibility confidence'' knob, a synthetic background depth, an artificial mask distance gradient, or a user-visible preflight blocker asking for a visibility policy.

The formal pose feasibility factor in the object-only setting is

$$
 \Phi_M(T,\mathcal P)=\begin{cases}0,&\text{every confirmed first-visible projected object point belongs to }\Omega^M_{kc}\text{ in every observed view;}\\ +\infty,&\text{otherwise.}\end{cases}
\qquad\text{(M5)}
$$

This feasibility indicator is **not differentiable** at pixel-bin changes. Apply it to trial poses **outside** the filter's Gaussian measurement Jacobian; it is a safeguard for the posed model, not an invented Gaussian mask measurement. The covariance $P$ of an accepted Gaussian filter remains a local conditional approximation and does not describe the full hard-truncated posterior near mask boundaries. If current poses are infeasible, propose new data-driven correspondences / initial rigid hypotheses in the outer loop; neither filter should silently bypass feasibility or exit during preflight.

## M4. IEKF observation Jacobian and sparse information (pipeline A ONLY)

Use world/spatial left SE(3) pose perturbations $T^+=\operatorname{Exp}(\widehat{\delta\tau})T$ with translation-first $\delta\tau=(\delta\rho,\delta\phi)$. Define $[v]_\times w=v\times w$. From (M3),

$$
 J_{3D}=R_{cw,c}[I_3,-[x_{ik}]_\times],\qquad
 J_\pi(X)=\begin{bmatrix}f_x/X_z&0&-f_xX_x/X_z^2\\0&f_y/X_z&-f_yX_y/X_z^2\end{bmatrix}.
\qquad\text{(M6)}
$$

The derivative of the **predicted observation** (M4) w.r.t. $(\delta\tau,\delta\nu)\in\mathbb R^{12}$ is

$$
\boxed{H_a=\begin{bmatrix}
 J_\pi J_{3D}&0_{2\times6}\\
 e_3^\top J_{3D}&0_{1\times6}\\
 0_{3\times6}&0_{3\times6}
\end{bmatrix}.}
\qquad\text{(M7)}
$$

Velocity/twist columns are zero **instantaneously** but are estimated through $P^-_{T\nu}$ and temporal motion propagation. For fixed prior $(\bar x^-,P^-)$ and IEKF iterate $\delta_\ell=\bar x_\ell\boxminus\bar x^-$ (including correct manifold chart transport), accumulation is

$$
\boxed{A_\ell=(P^-)^{-1}+\sum_{a\in\mathcal E}H_a^\top R_a^{-1}H_a,\qquad
b_\ell=\sum_{a\in\mathcal E}H_a^\top R_a^{-1}r_a-(P^-)^{-1}\delta_\ell,\quad A_\ell\Delta=b_\ell.}
\qquad\text{(M8)}
$$

Keep prior fixed while iterating and relinearizing the same observation. Apply an SE(3) retraction; use nonlinear conditional MAP acceptance and mask feasibility checks. Accumulate $12\times12$ group information blocks by GPU segmented reductions. If shared point/source-pose uncertainty is retained, use its sparse cross-factors/Schur complement, not an enormous independent point-state vector. Numerical damping is excluded from the *statistical* posterior covariance. The reference source's Gaussian $H^GJ_g$ is replaced **in its entirety** by (M7); alpha JVP/VJP does not survive the output-model substitution.

## M5. UKF nonlinear reprojection and low-rank update (pipeline B ONLY)

The UKF has the **same** state $(T,\nu)$, raw measurements (M4), masks (M1), associations, and outer loop, but does not use analytical measurement Jacobians. For $n=12$ dimensions per group, a nonaugmented update uses $L=2n+1=25$ state sigma points (augmented process prediction can have $2(n+q)+1$ sigma points). Retract $\delta\tau^{(j)}$ onto $SE(3)$ and directly evaluate each $h_a(x^{(j)})$ through (M3)--(M4). Keep the current raw observation IDs and mask gate fixed across the **conditional** sigma-point moment calculation. If visibility/association changes among sigma points, the chosen matched-point model is discontinuous: return the factor to the outer association stage rather than silently treating inconsistent sigma measurements as a smooth Gaussian vector.

For declared positive semidefinite covariance weights $w_j^c\ge0$, define $\bar h_a=\sum_jw_j^mh_a(x^{(j)})$, whitened sigma-output columns $Z_a(:,j)=\sqrt{w_j^c}(h_a(x^{(j)})-\bar h_a)$, and state sigma-deviation matrix $A_s(:,j)=\sqrt{w_j^c}(x^{(j)}\boxminus\bar x)$. With independent nonoverlapping raw matched pixel factors, the $L\times L$ reduction is

$$
\boxed{G=I_L+\sum_a Z_a^\top R_a^{-1}Z_a,\qquad
q=\sum_a Z_a^\top R_a^{-1}(z_a-\bar h_a).}
\qquad\text{(M9)}
$$

Then the **exact Woodbury implementation of this sigma-defined approximation** is

$$
\boxed{\delta x=A_sG^{-1}q,\qquad P^+=P^--A_s(I_L-G^{-1})A_s^\top.}
\qquad\text{(M10)}
$$

No $m\times m$ innovation covariance is built for $m$ measured pixel features; the small solve is $25\times25$ for an independent group. With correlated raw measurements, replace the diagonal block $R^{-1}$ sums with a shared sparse/raw-noise inverse operator. Predict nonlinear motion with augmented sigma points; save sigma cross-time covariance for **unscented** backward smoothing. Do not call IEKF or build a combined IEKF/UKF posterior. Evaluate candidate pose feasibility using exactly the same mask check as the IEKF before committing the accepted mean/covariance. This is a conditional UKF Gaussian approximation, not an exact Bayesian hard-constraint posterior.

## M6. Bidirectional solved/unsolved progression, common for both pipelines

1. Initialize `SOLVED` exclusively from $\Omega^D_{0c}$ across all cameras. Fuse consistent cross-camera observations with source-provenance and shared-pose uncertainty. Group membership is initially provisional; identity rigid transformation is an ordinary group hypothesis, not a special static branch.
2. For $k=1,\dots,K$: use the selected pipeline to predict $(T_g,\nu_g,P_g)$; transport solved points; project and match into $\Omega^D_{kc}$; apply only admissible raw-observation corrections; assign unmatched $\Omega^D_{kc}$ pixels to `UNSOLVED[k]`.
3. Apply **IEKF RTS** in A, **sigma-cross-covariance unscented RTS** in B, to the same forward trajectories. Do not treat smoothing as new image evidence.
4. Visit late-to-early `UNSOLVED[s]`: initialize candidate point from the *real foreground depth* in its source frame. For any group candidate, transport between frame $s$ and earlier $k$ by $T_{gk}T_{gs}^{-1}q_s$; search foreground correspondences and respect depth/mask consistency. Infer new motion groups only from sufficiently informative coherent multiframe correspondence and the declared model complexity penalty. If insufficient, leave observations unresolved rather than fabricate a group trajectory.
5. Accept map insertion/fusion **transactionally**, preserving the same canonical group frame and source IDs; reject candidate changes if the same full objective worsens. Incorporate pose--point and cross-point uncertainty induced by common uncertain source poses.
6. Progressively revisit frames $s=1,\dots,K$ and their remaining UNSOLVED observations against earlier frames, update affected trajectories using **only the selected filter**, then perform a second full forward sweep with the expanded map. Repeat until accepted model improvements cease; do not assert every observation is solvable.

For source observation $q_s$ backprojected at $s$, the canonical coordinate under group $g$ is $p_i=T_{gs}^{-1}q_s$, and the predicted earlier point is

$$
\boxed{\hat q_k=T_{gk}T_{gs}^{-1}q_s.}
\qquad\text{(M11)}
$$

Source-frame pose uncertainty makes different newly initialized points correlated even when their RGB-D pixel noise is independent. Marginal covariance cannot be represented by independently averaging $3\times3$ point covariances. Keep sparse source-pose factors/Schur elimination or explicitly report a **conditional** covariance. Do not use ground-truth joint/part values for reconstruction.

## M7. Sparse GPU implementation and exact memory contracts

Use native frame+camera-major contiguous RGB (`H,W,3`), depth (`H,W`), mask (`H,W`, byte), validity (`H,W`, byte), and immutable raw observation IDs. Persistent `p[N,3]`, `rgb[N,3]`, `group[N]`, `source[N]`, `pose[G,4,4]`, and optional sparse covariance factor metadata are *only the currently solved map*, not the $55$ million raw observations duplicated as map states. Stream frames/cameras or decode into a bounded cache.

| Kernel / stage | GPU input --> output | Invariant |
|:--|:--|:--|
| `compact_valid_fg` | `(M,V,D)` --> sorted raw valid indices | exact $M\land V$, no morphology/epsilon |
| `backproject_fg` | valid IDs, `K^-1,D,Twc` --> XYZ, RGB | source IDs preserved; initialize frame 0 only |
| `transform_project` | grouped points, poses, camera --> `(uv,z,pixel_bin,valid)` | one rigid transform per group; no dense point-pixel tensor |
| `zbuffer_first` | valid projected triples --> `(min_z,pointID)` per bin | deterministic depth+ID ordering, bounded scratch |
| `associate_raw_fg` | projection, mask/depth/RGB, sparse candidates --> CSR edges | no background targets, unique raw-pixel primary ownership |
| `reduce_iekf` | CSR edges, analytic $H,r,R^{-1}$ --> group-block `(12,12),(12,)` | A only; deterministic/reference reduction parity |
| `evaluate_sigma_ukf` | same CSR IDs, 25 poses per group --> streamed $(Z^TR^{-1}Z,Z^TR^{-1}r)$ | B only; sigma visibility support consistency |
| `reverse_correspond` | late unsolved valid IDs and relative group transforms --> earlier CSR matches | same projector, same masks; no 3DGS renderer |
| `fuse_points` | accepted matched raw IDs --> canonical point/map factor edit | one physical point ID; transactional rollback |

Typical active GPU memory is $O(N_{solved}+CHW+E+G\,12^2)$ for A or $O(N_{solved}+CHW+E+G\,L^2)$ for independent-group B (plus bounded sigma projection scratch and sparse point--pose factors). Group-coupled covariance has dimension $12G$ and may be sparse/block structured; do not promise independent $12\times12$ blocks if the stored state couples groups. In B, process sigma point count for one 12D state with 12D noise augmentation is $49$, and update count $25$; do not allocate 25 complete rendered images over all frames. On 8GB GPUs use chunked projection, compact native `float32` geometry and `float64` small matrix accumulation/solves where needed. No full-scan repeated traversal of every point for every raw pixel; no all-frame all-observation proposal catalogue.

**GPU numerical contract.** Warp-safe associative sums or two-pass segmented reductions; test deterministic/reproducible mode against a CPU FP64 reference. Color scale, depth units, camera optical axis, pixel-center convention, and SE(3) tangent convention must agree between CPU and GPU. Run explicit memory-budget preflight using *real current scene sizes*, without using arbitrary cap-based geometry subsampling to make an invalid design appear executable. Cache only when the key includes frame, camera, point-map version, pose version, mask/validity version, and association version; invalidate z-buffers and residuals on accepted state/map edits. No stale objective comparisons.

## M8. Production tests and observation-model audit

The following are **mandatory in both branches**, on top of the preexisting IEKF/UKF theory tests:

1. On the source 3DGS PDF Eq. (41), verify code equivalence $P=MI$ and $V^{D,used}=V^{raw}\land M$ pixel by pixel; verify no background-depth precision or adjoint.
2. Parse binary masks 0/1 and 0/255 without thresholding; detect mismatched mask/depth/RGB dimensions and camera IDs.
3. Perturb every mask-zero RGB and depth value while keeping foreground pixels unchanged: all point targets, associations, updates, and discoveries are **identical** (under the object-only assumption stated in M3).
4. With $M=0$ everywhere, no persistent point can initialize or be discovered, and no filter observation factor may be added.
5. A point frontmost under the declared object-only premise and projected at $M=0$ makes a trial infeasible; a point behind a nearer valid foreground observation remains solved and has no direct update.
6. Same-depth predictions use the declared stable `(depth,pointID)` ordering. Out-of-frame points impose no mask requirement; foreground invalid depth yields no point observation.
7. Unique raw ID accounting: $|\Omega^D|=|\mathrm{OWNED}|+|\mathrm{UNSOLVED}|$ per frame and camera; duplicate cameras/frames never silently clone the same physical map point.
8. Finite-difference $(\hat u,\hat d)$ Jacobians for IEKF; pose/twist process $F,G$ and covariance chart tests; no camera-space/world-space sign swaps.
9. UKF sigma moments and Woodbury `25x25` match an explicit small dense innovation reference, including posterior mean/covariance; UKF predictive cross-covariance matches linear RTS on linear fixtures.
10. If bilinear photometric mode exists, check two overlapping sampled pixels produce the correct off-diagonal covariance $W R_{raw} W^T$; never silently use independent per-sample color noise.
11. Test reverse transport $T_kT_s^{-1}$ and a late point hidden behind valid earlier foreground depth; it remains feasible and is not inserted twice.
12. Test a once-only visible point remains unresolved when its group identity cannot be established; no forced static or moving guess.
13. Verify cache invalidation on trial pose/map/association modifications, including trial rollback and independent branch data copies.
14. Run **both** programs on every frame and all cameras; demand forward filter estimates, own backward smoother, real backward map additions, chronological frontier resolution, and a second complete forward pass. Export objective/feasibility history, accepted map-point IDs, unknown observations, pose trajectories, covariance validity and GPU peak memory. Component-test success alone is not reconstruction.
15. Fail any run that calls 3DGS alpha, random-background compositing, surface normals, a separate visibility confidence rule, or an eager all-frame point catalogue in the production path.

**Limits to report.** The 3DGS background-expectation theorem is exact for its specified alpha-composited Gaussian quadratic cost, not for the point-based output. The point mask check is a discrete feasibility rule, not a smooth RGB residual. A conditional IEKF and a sigma-point UKF are both approximate nonlinear Bayesian estimators; no one-pass global convergence or complete reconstruction theorem follows. New group identification and unobservable points may remain unresolved. A decrease of one frozen-association conditional score is not proof that the full changing-map objective decreased.


# COMMON MATHEMATICAL CONTRACT — RGB-D Solved/Unsolved SLAM Outer Loop

**Status:** Normative, shared without modification by the standalone IEKF and UKF implementations. No hybrid filter, no filter selection, no 3DGS, no implicit field, no surface normals, no separate static/dynamic pipeline.

## C0. Inputs and geometric variables

Input: synchronized calibrated multiview RGB-D observations at acquisition times `t[0:K+1]`:

$$
\mathcal Y_k=\{(I_{c,k},D_{c,k},M_{c,k},K_c,T_{cw,c})\}_{c=1}^C.
$$

`M` is a foreground observation-domain mask, NOT a motion or part label. Respect the actual selected scene's timestamps/FPS; `dt_k=t_{k+1}-t_k>0`. Depth is metric and invalid/no-return pixels have a separately documented observation model. Use all frames and all cameras for a production run; subsampling is allowed only in explicitly named diagnostic tests.

Backprojection of a valid depth pixel `u=(u_x,u_y)` into world coordinates:

$$
q_{kcu}=T_{wc,c}\,D_{c,k}(u)K_c^{-1}[u_x,u_y,1]^\top,
$$

with appropriate homogeneous coordinates. Raw observation IDs `(k,c,u)` are immutable.

At any outer iteration maintain ONLY two conceptual sets:

- `SOLVED`: persistent canonical 3D points `p_i`, fixed or revisable RGB `a_i`, originating observation provenance, uncertain rigid-group membership and map covariance/information references. A point is `solved` if assigned a currently accepted geometric/rigid-motion explanation. Initially the fused points observed at `k=0` populate `SOLVED`; their motion group memberships are provisional because frame 0 alone cannot identify motion.
- `UNSOLVED[k]`: raw foreground RGB-D observation IDs that the currently accepted explanation does not claim. These are NOT persistent state variables. All observations remain on disk and may be revisited; only a compact index, flags and local hypotheses are retained.

The groups are indexed `g` with shared poses `T[g,k] in SE(3)`; the number of groups may change under accepted outer-loop model edits. An identity trajectory `T[g,k]=I` describes the stationary case **without any distinguished static class or static-specific codepath**. A group can be parametrized relative to any reference frame `s_g` through `T[g,s_g]=I`; transform its canonical coordinates if its gauge is changed. No ground-truth joint type, joint angle, motion label or part segmentation participates in inference.

The sole predicted position of solved point `i` at `k` is

$$
x_{i,k}=T_{g(i),k}p_i,
\qquad
x_{i,k}=T_{g(i),k}T_{g(i),s}^{-1}q_{i,s}\quad\text{for a point acquired at }s.
$$

This relative transport formula is IDENTICAL forward and backward. Store no independently moving trajectory or velocity per point.

## C1. Direct RGB-D measurement and sparse association

For a solved point `p_i`, given a candidate pose `T[g,k]`, transform to camera coordinates `X_{ick}=T_{cw,c} T[g,k] p_i`, project `u_hat=pi_c(X)`, predicted depth `d_hat=e_3^T X`, and predicted RGB `a_i`.

**Canonical matched-observation model:** associate `i` with an *actual* target raw pixel `u` using a one-to-one or explicitly many-to-one sampling-aware matching model:

$$
z_{kcu}=\begin{bmatrix}u_x&u_y&D_{c,k}(u)&I_{c,k}(u)^\top\end{bmatrix}^\top,
\qquad
h_{ick}(T,p_i)=\begin{bmatrix}\pi_c(X)^\top&e_3^\top X&a_i^\top\end{bmatrix}^\top.
$$

`r=z-h`, with correctly configured covariance `R_{kcu}` for pixel localization / association, depth and RGB. The image coordinates are measurements of the associated target location; they are not arbitrarily interpolated depth. The RGB part informs association and/or the shared point's appearance; at fixed association and frozen appearance it has zero pose derivative. For direct photometric alignment instead, use `I_k(pi(X))-a_i` with an analytical image-gradient Jacobian and **propagate correlations between overlapping bilinear samples** from the original raw pixels. Do not count both forms as independent observations of the same RGB pixels.

Matching is local: project predicted solved points to each camera, use pixel/tile bins and a depth-buffer/order test, query compatible observed pixels in the projected support, and solve deterministic local assignment with an explicit `unassigned` alternative. Use the actual RGB/depth/pixel uncertainty and a declared model prior. No hand-chosen norm-based novelty threshold. Keep raw observation IDs so one measured pixel is not incorporated independently several times. Simple point samples and pixel quantization are adequate for the matching objective; **normals, TSDFs, surfels, Gaussian rendering and differentiable splatting are NOT required**.

A point sample does not reconstruct a continuous image everywhere. Interpret predicted RGB-D images as **partial sparse projections** with valid/unknown masks and z-buffering; do not claim dense-pixel likelihoods for uncovered pixels. Unexplained measured pixels stay in `UNSOLVED`. A full generative image model would require specifying pixel support; do not silently assume one.

## C2. Exact solved/unsolved outer schedule

### Initialization
1. Load cameras, masks, metric depth and source timestamps/FPS. Determine RGB-D and process-noise assumptions from the authorized existing configuration; distinguish assumptions from independent calibration.
2. Backproject and fuse overlapping **frame-0** observations across cameras through local correspondence, constrained by uncertainty and matching evidence. Do not concatenate millions of duplicates into permanent map variables.
3. Set `SOLVED <- frame0_fused_points`; set `UNSOLVED[k>0] <- all valid masked observations`; use frame-0 correspondence provenance to avoid treating the same measurements again as independent prior evidence.
4. Bootstrap rigid-group motion **from observed interframe correspondences**. Start with the fewest supported motion hypotheses, but do not freeze all frame-0 points into one un-splittable body. A persistent residual subgroup can propose a split; accept/reject through the common scoring rule in C5. No preassigned moving/static pattern.

### Forward tracking sweep
For `k=1..K`:
- Predict poses/twists and their uncertainty from the preceding state (IEKF or UKF according to the standalone implementation).
- Move only the points currently in `SOLVED` using common group transformations. Project, depth-order and find correspondences in observed RGB-D point measurements.
- Run the **inner filter for matched, supported group states only**. If correspondence structure changes, update it externally between inner solves, not by storing all pixels in the state.
- Mark matched measured observations as explained; leave others in `UNSOLVED[k]`. A point that disappears because it is behind measured foreground is retained in `SOLVED` but simply receives no surface observation update while occluded.
- Save filtered state, predicted state, associated observations, uncertainty and evidence needed for backward smoothing.

After `k=K`, smooth the SAME tracked motion trajectory backward via the filter-specific smoother. This alone does not discover new geometry.

### Late-to-early discovery sweep
For `s=K..1`, consider coherent local groups of `UNSOLVED[s]`, **not an eager permanent map entry for each pixel**:
- Test transporting candidate measured 3D points backward by each currently solved group trajectory `T[g,k] T[g,s]^{-1}`. Include a new rigid-group hypothesis if existing groups fail to account for joint cross-frame evidence. Identity trajectories are allowed automatically as normal poses.
- Use sparse backward projective matching and visibility with earlier RGB-D images. Behind a nearer depth measurement = possible occlusion, projected into measured free space = contradiction, off-image / invalid reading = uninformative. Evaluate with declared uncertainty, not an arbitrary fixed residual cutoff.
- Where several raw observations describe the SAME physical 3D point under one consistent rigid motion, fuse them into ONE canonical point with source provenance and covariance/cross-information. Missing or weakly constrained observations remain `UNSOLVED`.
- Optimize proposed new group motions using **the same chosen inner filter** operating in reverse observation order or by using its filter-specific temporal smoother; never create a second unrelated trajectory. Proposal construction may use deterministic weighted rigid registration as an initializer, not ground-truth labels.
- Accept a new point, group or re-assignment ONLY when the shared full-sequence score including unmatched likelihood and model complexity improves. Until accepted, keep edits provisional (transactional map edit).

### Progressive chronological frontier
For `s=1..K`:
- Reproject all currently solved geometry at frame `s`; recompute matched/unmatched observation IDs.
- For `UNSOLVED[s]`, search backward through `s-1..0` for consistent relative rigid transport, including existing-group and new-group hypotheses. Add accepted coherent geometry to `SOLVED` and revise the affected group trajectories.
- Regenerate `UNSOLVED` membership from observation ownership; no cumulative single-use 'discarding' of an observation.

Repeat: forward filter -> filter-specific backward smoothing -> late-to-early discovery -> progressive frontier until no accepted edits or meaningful accepted improvement remains under the fixed numerical optimizer precision and unchanged scoring definition. Never promise all observed points become solved: some observations are fundamentally unidentifiable or noisy.

## C3. No ad-hoc status threshold: deterministic decisions via a declared probability model

Every candidate association, unassigned observation, rigid group and fused point must be evaluated with a consistent negative log posterior:

$$
J(\mathcal S,\mathcal G,\mathcal A,\mathbf T,\mathbf p)=
\sum_{\text{owned raw observations}}[-\log p(z_a\mid\text{assigned point, pose})]
+\sum_{a\in\mathcal U}[-\log p_{\mathrm{unassigned}}(z_a)]
+J_{\mathrm{motion\ prior}}+J_{\mathrm{map\ prior}}-\log p(\mathcal G,\mathcal A).
$$

All model probabilities and priors must be explicitly specified/provenanced, normalized where required, and used in both pipelines. If the user supplied no prior or a parameter is genuinely not identifiable, report it. A standard observation likelihood permits a deterministic MAP decision (including `unassigned`) without a hand-coded distance threshold. **The exact combinatorial global association optimum is generally computationally intractable; local sparse assignment and group proposals are algorithmic restrictions, not magically exact global Bayesian inference.** Identify the optimization class and record accepted candidates.

A proposal is accepted if its score with the new model is lower than the same score with the old model; numerical tolerances relate only to solve accuracy. Unfavorable proposals are rolled back. Never count a source-frame observation as both an independent landmark prior and a new measurement.

## C4. Geometry uncertainty and shared motion uncertainty

For `p=T_{g,s}^{-1} q_s`, initialize `Cov(p)` using Jacobians w.r.t. RGB-D observation and pose:

$$
P_{pp}=J_q R_q J_q^\top + J_T P_{TT}J_T^\top
+\text{correlation cross terms (if present)}.
$$

Also store `P_{pT}` or sparse information factors; two new points sharing an uncertain source pose are correlated. Do not count them as independent motion constraints. Only the **conditional inner-motion filter** may operate on a small pose/twist covariance; joint map-motion marginal uncertainties require selected sparse covariance queries or marginalization with cross terms. Gauge fix one reference rigid pose for each canonical group chart; do not invert a singular full pose/map gauge. For new group or first observed point, retain a conditional prior, not falsely precise initial pose or identity-motion certainty.

Use a well-defined initial `P0`, `Q(dt)`, sensor `R`, and output-no-return/association model; read FPS/timestamps from actual input metadata. Approved numerical guesses from previously supplied configuration are assumptions and must be logged with provenance. Do not invent unrelated constants and call them measurements.

## C5. GPU and memory invariants

- Store `N_map x 3` canonical XYZ, RGB and group IDs and small covariance/information metadata; `UNSOLVED` stores indexed raw observations or compact per-frame bitsets.
- Camera rays and observations are **streamed one frame/batch at a time**. No `N_frames*N_cameras*N_pixels*N_groups` persistent tensor and no cloned XYZ per group.
- Project and bin only solved visible points; candidate discovery operates on current unresolved patches and nearby frames, streaming observations.
- Each accepted data association contributes an `O(1)`-width Jacobian of the relevant motion state, not a per-pixel state variable.
- Per frame memory `O(N_map+N_active+N_raw_frame+12^2 B^2)`, plus optional stored `O(K(12B)^2)` smoother covariance. Point/map cross-information is retained sparsely or via selected solves, never dense `N_map x N_map`.
- GPU kernels: batched `SE(3)` point transport, calibrated projection, sparse image-space matching, depth-order/visibility, segmented reductions or unscented low-rank moments, and optional deterministic rigid alignments.
- Respect physical RGB-D mask domain; do not treat entire unmasked environment as object geometry.

## C6. Shared outputs and mandatory comparable experiments

Each implementation outputs identically named *semantic* products under separate directories:

`solved_points.ply`, `solved_points.csv` (point IDs and group IDs), `unresolved_observations.csv` (frame/camera/pixel IDs and reasons/probabilities), `group_poses.npz`, `group_twists.npz`, `group_covariance.npz` (appropriately labeled conditional/marginal), `match_log.csv`, `stage_log.csv`, `objective_history.csv`, `forward_depth/`, `forward_rgb/`, `backward_depth/`, `backward_rgb/`, `run_config_resolved.yaml`, `validation_report.json`.

Both implementations MUST read the exact same dataset and matching/noise config, start with the exact same frame-0 fused points, use identical association proposals and objective definitions, and differ ONLY in their inner filtering and temporal smoothing. Do not favor one by using different sensors, segmentation, initialization, or GPU budgets.

### Acceptance criteria

1. No normals, surfels, TSDF, Gaussian renderer or dense point/observation state in either code path.
2. Full scene (all actual timestamps/cameras) loaded and used; masks, camera conventions and metric depths tested.
3. Frame-0 initialization uses only frame 0; source observations are not counted twice.
4. Successful measured forward tracking; nonidentity motion not artificially suppressed.
5. At least one real backward discovery attempt with actual source and earlier frame indices; new points fused with provenance if evidence permits.
6. Progressive frame-frontier pass revisits remaining unsolved observations.
7. Pose state covariance has correct process and observation updates; back smoother returns smoothed pose AND covariance.
8. Actual reconstructed RGB/depth projections exported with valid/unknown masks; no fabricated dense image coverage.
9. Objective must be evaluated on one unchanged likelihood/prior across outer updates; accepted stage values nonincreasing.
10. No `3**N`, no fixed whole-scene assignment patterns, no dense full-measurement covariance, no arbitrary per-observation permanent map variables.
11. Full run completes cycles; numerical tests alone or a first objective evaluation are not production success.
12. Synthetic occluded/delayed-appearance and motion-group-split tests recover the correct groups without ground-truth inference leakage.

### Explicit mathematical limitations

Matched-point correspondence is a discrete variable and rigid motion alone may be unobservable from one point or symmetric geometry. Neither IEKF nor UKF guarantees the globally optimal association or complete reconstruction of never-observed geometry. Monotonic accepted posterior scores do not prove convergence to the true physical scene.


\newpage

# FRAMEWORK A — Analytical Iterated Error-State EKF and RTS RGB-D SLAM

**Independent theory.** The entire solved/unsolved reconstruction algorithm is *exactly* `00_COMMON_SOLVED_UNSOLVED_SPEC.md`. This document derives its sole inner motion solver: **analytical IEKF**. It does not call a UKF or dynamically switch filters.

## A1. Problem, conditional measurements and Bayesian state

At frame `k`, let there be `B` currently accepted rigid groups. The conditional Gaussian motion state for group `g` is

$$
s_{g,k}=(T_{g,k},\nu_{g,k}),\quad T_{g,k}\in SE(3),\quad \nu_{g,k}\in\mathbb R^6.
$$

Its 12D error state is `delta s=(delta xi, delta nu)`. The full frame error state dimension is `n=12B` (not a function of the number of points). `SOLVED` contains fused persistent point IDs and their accepted or provisional motion associations. `UNSOLVED[k]` is solely an indexed set of raw observations. A static trajectory is simply `T[g,k]=I`; no static label is a state variable.

For translation-first `xi=(rho,phi)` define `xi^ = [[phi]x,rho;0,0]`, `T=Exp(delta xi^) Tbar`, `nu=nubar+delta nu`. The gauge is a group's reference frame `T[g,s_ref]=I`; its point coordinates are in that frame. A truly static group need not be predeclared and its estimated motions may approach identity naturally.

Conditional on an external association `a=(i,k,c,u)` with a solved canonical 3D point `p_i`, the observed vector is

$$
z_a=(u_x,u_y,D_{c,k}(u),I_{c,k}(u)^\top)^\top\in\mathbb R^6,
\quad
z_a=h_a(s_k,p_i)+\epsilon_a,
$$

$$
h_a=\big(\pi_c(X_a)^\top,\;[X_a]_z,\;a_i^\top\big)^\top,
\quad X_a=R_{cw,c}(R_{g,k}p_i+t_{g,k})+t_{cw,c},\quad
\epsilon_a\sim\mathcal N(0,R_a).
$$

`R_a` describes target-pixel association/localization, depth and RGB uncertainty, and its cross terms where applicable. A matched-pixel RGB residual is an **association / appearance** factor: with point appearance fixed, its pose derivative is zero. If true direct photometric pose refinement is wanted, additionally/alternatively use warped-image residual `I_k(pi(X))-a_i` with analytical `nabla I` and covariance propagated from raw pixels, not a second independent reuse of the RGB pixel. There are no normals or explicit surface elements. The sparse projector applies the deterministic foreground-only projection and conditional depth-order procedure (M3), without any configurable visibility guard. Nonmatching observations stay in `UNSOLVED` and do not generate fake residuals.

**Meaning of exactness:** `H` is the exact analytical derivative of the *specified conditional* output; the iterated EKF is still a local Gaussian approximation to nonlinear Bayesian filtering. Correspondences are estimated outside the filter and may be ambiguous.

## A2. Nonlinear dynamic state equation and process covariance

Use spatial/world twist `nu=(v,omega)` and the same left perturbation chart throughout:

$$
\boxed{\begin{aligned}
T_{g,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{g,k})T_{g,k}\operatorname{Exp}(\widehat w^T_{g,k}),\\
\nu_{g,k+1}&=\nu_{g,k}+w^\nu_{g,k}.
\end{aligned}}
$$

`w=(wT,wnu)` has the documented model covariance `Q_k`. The right-applied pose process noise is mapped into the left error chart. Define `phi=Delta t * nubar`, `A=Exp(phi^)` and `Tpred=A Tbar`. Translation-first adjoint is

$$
\mathrm{Ad}_{(R,t)}=\begin{bmatrix}R&[t]_\times R\\0&R\end{bmatrix}.
$$

With the left group Jacobian defined by `J_l(phi) = integral_0^1 Ad_{Exp(s phi^)} ds`, the first-order state transition is

$$
\boxed{F_k=\begin{bmatrix}\operatorname{Ad}_{A}&\Delta t_k J_l(\phi)\\0&I_6\end{bmatrix}},\qquad
G_k=\begin{bmatrix}\operatorname{Ad}_{T_{pred}}&0\\0&I_6\end{bmatrix}.
$$

Compute (for all groups, including any cross-block covariance)

$$
\boxed{P^-_{k+1}=F_kP_k^+F_k^\top+G_kQ_kG_k^\top.}
$$

If using a continuous-time white-acceleration model, discretize its spectral-density covariance with the actual `dt`, including pose/twist cross terms. Do not replace it with independent diagonal position and velocity noise without model justification. `P0` and `Q` numerical assumptions must come from the approved existing configuration and be labeled as assumptions. The chosen gauge is fixed before inversion.

## A3. Analytical RGB-D output Jacobian

Let `x=R_g p_i+t_g` and `X=R_c x+t_c` in camera coordinates. With `e3=(0,0,1)^T`, `pi(X)=(f_x X_x/X_z+c_x, f_y X_y/X_z+c_y)`:

$$
J_{pose3}=R_c\begin{bmatrix}I_3&-[x]_\times\end{bmatrix},\qquad
J_\pi(X)=\begin{bmatrix}f_x/X_z&0&-f_x X_x/X_z^2\\0&f_y/X_z&-f_yX_y/X_z^2\end{bmatrix}.
$$

The conditional output derivative wrt `delta s=(delta xi,delta nu)` is

$$
\boxed{H_{a}=\begin{bmatrix}J_\pi J_{pose3}&0_{2\times6}\\e_3^\top J_{pose3}&0_{1\times6}\\0_{3\times6}&0_{3\times6}\end{bmatrix}}\in\mathbb R^{6\times12}.
$$

For the alternative *warped* RGB residual `r_RGB=I_k(pi(X))-a_i`, its pose derivative is `grad I_k(pi(X)) J_pi J_pose3` (sign depends on whether Jacobian is defined for `r` or `h`). For source-acquired points `T_k T_s^{-1} y_s`, both `T_k` and `T_s` contribute analytically when jointly refined: differentiate the relative transform rather than freezing the uncertain source pose. For `p_i` uncertainty, `J_p = R_c R_g`; incorporate its conditional residual covariance or retain sparse point-pose information factors.

No finite differences are permitted in the production analytical Jacobians. Verify them by finite differences in tests, including camera world/convention transforms and left chart perturbations.

## A4. Prediction, iterated correction and covariance

For the fixed predicted state `(s^-,P^-)`, build the deterministic current correspondence and visibility set outside the IEKF. At iteration `ell`, define `delta_ell = s_ell boxminus s^-`, observed residual `r_ell=z-h(s_ell)` and current `H_ell = dh/d delta` (with chart transport of prior terms if the chart changes).

For a fixed linearization chart the Gauss-Newton form of the IEKF correction is

$$
\boxed{\left[(P^-)^{-1}+H_\ell^\top R^{-1}H_\ell\right]\Delta_\ell
=H_\ell^\top R^{-1}r_\ell-(P^-)^{-1}\delta_\ell.}
$$

The prior `(s^-,P^-)` is **fixed for all inner relinearizations**. The measurement is used once even if a 10-step IEKF is used to converge. Apply a Lie retraction and optionally backtracking against the *actual fixed-association conditional MAP objective*, then reproject and relinearize. External matching changes may trigger a new well-defined conditional problem, but a trial update must not compare scores from different matches without invoking the same global objective.

Measurement information can be accumulated without constructing `H`:

$$
A=(P^-)^{-1}+\sum_{a\in E_k}H_a^\top R_a^{-1}H_a,\quad
b=\sum_{a\in E_k}H_a^\top R_a^{-1}r_a-(P^-)^{-1}\delta.
$$

For one group `A` is `12x12`; cross-group factors produce sparse `12x12` blocks. For a joint state with map variables, expose `Hv` / `H^T v` or Schur-eliminate the sparse map blocks. No dense `N_points x N_points` or `N_pixels x N_pixels` arrays. Uncertainty **after** the accepted update must be computed with the undamped local posterior information `A`, with correct tangent-chart transport. Levenberg damping used for step acceptance must not artificially increase posterior precision. A Joseph-form covariance update with the equivalent linearized `(H,R)` is a small-case reference test. Do not call this exact marginal covariance when map uncertainty and point-pose coupling are omitted.

## A5. Backward motion smoothing (IEKF branch only)

After one forward filtered run save `(s^+_k,P^+_k)`, `(s^-_{k+1},P^-_{k+1})` and `F_k`, with reference tangent frames. Compute Rauch-Tung-Striebel gain

$$
C_k=P^+_k F_k^\top(P^-_{k+1})^{-1}.
$$

Using tangent errors expressed in the appropriate charts:

$$
\delta_k=C_k\big(s_{k+1|K}\boxminus s^-_{k+1}\big),\quad
s_{k|K}=s^+_k\boxplus\delta_k,
$$

$$
P_{k|K}=P^+_k+C_k(P_{k+1|K}-P^-_{k+1})C_k^\top,
$$

with coordinate transport to the updated tangent. This is the first-order Lie-group RTS smoother; it does not discover points. For map poses correlated with geometry, the small conditional state smoother must not be labeled the exact joint map marginal smoother: use retained sparse joint factors or selected Schur-complement queries.

## A6. SAME solved/unsolved outer loop, no filter mixing

Use the algorithm in `00_COMMON_SOLVED_UNSOLVED_SPEC.md` **without modification**:

1. Frame-0 multiview points form `SOLVED` (group affiliation initially provisional); all other valid RGB-D observations start in `UNSOLVED`.
2. Forward for every `k`: predict pose/twist + covariance, move only solved points, sparse associate RGB-D, run analytic IEKF correction, retain unmatched observations.
3. Backward RTS motion smoothing of the *same* poses/covariances.
4. Late-to-early reverse discovery: transport unsolved points under each accepted group `T[g,k] T[g,s]^{-1}`; compare backward visibility, RGB and depth. Propose new group only from coherent multiframe correspondence; its initial rigid fit is a deterministic initializer for IEKF.
5. Fuse accepted points into one canonical point per physically matched location, retaining provenance and shared pose uncertainty.
6. Progressive chronological frontier: for `s=1..K`, solve `UNSOLVED[s]` against earlier frames, update affected trajectories via this IEKF, and refresh associations.
7. Repeat full outer sweeps while the common global objective actually improves.

No named static body, no normals, no UV/texture atlas, no TSDF, no dynamic 3DGS. No UKF fallback anywhere in this framework.

## A7. Convergence: valid statement, not an overclaim

For a fixed finite map, group structure and association set, if the measurement function is locally continuously differentiable with locally Lipschitz Jacobian, declared `R` and gauge-fixed prior are positive definite on the estimated tangent subspace, and an Armijo/damped IEKF line search ensures sufficient decrease, accepted conditional MAP objective values decrease and have a finite limit when bounded below. Under standard boundedness/nondegeneracy and sufficient-descent conditions, accumulation points are stationary for that *fixed-association local objective*. This does not establish global registration or scene discovery correctness.

The complete outer loop changes data association and point/group existence discretely. If every accepted outer change is evaluated on the **same comparable full sequence negative log-posterior including a proper unassigned-observation and model-complexity term**, its accepted values are nonincreasing and converge as scalar values when bounded below. Stationarity across discrete changes and guaranteed recovery of all occlusions do not follow. Outlier, invalid depth, missing cameras and genuinely invisible geometry may remain `UNSOLVED`.

## A8. Computational bounds and falsifiable tests

- Runtime: roughly `O(sum_k active_matches_k * C_pose + K * solve(12B))` per sweep, plus sparse GPU matching and backward point-discovery passes.
- Memory: `O(N_solved + max_raw_frame + K (12B)^2 + sparse_map_factors)`, not `O(K*C*W*H*B)` persistent state.
- Required tests: left-chart `F,G` vs finite differences; `H` vs finite differences; correct predicted depth and camera conventions; IEKF vs linear Gaussian analytic posterior; Joseph/reference covariance; fixed prior across inner iterations; RTS vs dense batch reference on small problem; shared pose covariance after late-point insertion; both temporal directions; empty/occluded/no-return behavior; group splitting from initial frame geometry; long-sequence complete forward-backward-frontier loop; monotone global score for every accepted outer edit.

**Deliverable:** Theory-compliant standalone analytical IEKF motion module, its own RTS smoother, and the shared solved/unsolved outer algorithm. All outputs go to `results_iekf/` and are labeled **IEKF** only.


\newpage

# FRAMEWORK B — Independent Lie-Group Unscented Kalman Filter and Unscented Smoother for RGB-D SLAM

**Independent theory.** Use precisely the solved/unsolved outer algorithm defined in `00_COMMON_SOLVED_UNSOLVED_SPEC.md`. In this version the inner motion estimator is **only an augmented Lie-group UKF**; the backward temporal pass is **only an unscented RTS smoother**. There is NO IEKF fallback, analytical measurement-Jacobian step, or hybrid selector.

## B1. Same observations, state and geometry as Framework A

At each time `k`, with `B` currently supported rigid groups, use

$$
s_k=\{(T_{g,k},\nu_{g,k})\}_{g=1}^B,\quad T_g\in SE(3),\quad \nu_g\in\mathbb R^6,\quad \dim(\delta s_k)=n=12B.
$$

The canonical map `SOLVED` consists of persistent XYZ/RGB point samples and their rigid-group ownership. `UNSOLVED[k]` holds raw observation IDs not currently explained. They are NOT components of the UKF state. All transformations may be identity; there is no special static label or separate static-motion estimator.

Use the same translation-first tangent chart as the IEKF derivation:

$$
s\boxplus(\delta\xi,\delta\nu)=(\operatorname{Exp}(\widehat{\delta\xi})T,\nu+\delta\nu),\quad
s_1\boxminus s_2=(\operatorname{Log}(T_1 T_2^{-1})^\vee,\nu_1-\nu_2).
$$

The state output for a matched raw RGB-D point observation is identical to Framework A:

$$
z_a=(u_x,u_y,D_{c,k}(u),I_{c,k}(u)^\top)^\top,
$$

$$
h_a(s_k,p_i)=(\pi_c(X_a)^\top,[X_a]_z,a_i^\top)^\top,\quad
X_a=T_{cw,c}T_{g,k}p_i.
$$

Raw localization/depth/RGB observation covariance `R_a` and association rules are **exactly the same**. A raw RGB residual informs correspondence and shared appearance, but fixed point color has no local derivative with respect to pose; the UKF likewise cannot infer pose from a state-independent RGB output. To use direct photometric pose evidence, substitute the explicitly specified warped-image output `h^photo=I_k(pi(X))`, and propagate shared raw-pixel measurement covariance; never count the same RGB sample twice.

## B2. Dynamic equation: nonlinear constant-twist process

The physical dynamic model is IDENTICAL to Framework A:

$$
\boxed{\begin{aligned}
T_{g,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{g,k})T_{g,k}\operatorname{Exp}(\widehat{w^T_{g,k}}),\\
\nu_{g,k+1}&=\nu_{g,k}+w^\nu_{g,k}.
\end{aligned}}
$$

`w=(wT,wnu)` is an independent process-noise vector in the declared right-pose/linear-twist noise chart, with covariance `Q_k`, discretized from the same approved physical assumptions and actual `dt_k` as Framework A. **No analytical state-transition Jacobian is used for UKF prediction.**

### Augmented sigma-point prediction

Let the state error dimension be `n=12B`, process-noise dimension `q=12B` for the stated process. Construct an augmented zero-mean error `eta=(delta s,w)` with covariance

$$
P_{aug}=\operatorname{diag}(P_{k|k},Q_k)\in\mathbb R^{(n+q)\times(n+q)}.
$$

Choose and log valid sigma-point parameters `(alpha,beta,kappa)` with `n_a=n+q`, `lambda=alpha^2(n_a+kappa)-n_a`, `n_a+lambda>0`. Generate `2 n_a+1` deterministic sigma points with exact Cholesky (or PSD eigendecomposition if necessary), using

$$
\eta_0=0,\quad\eta_j=+\sqrt{(n_a+\lambda)P_{aug}}_j,\quad
\eta_{j+n_a}=-\sqrt{(n_a+\lambda)P_{aug}}_j.
$$

Weights: `wm0=lambda/(n_a+lambda)`, `wc0=wm0+1-alpha^2+beta`, other weights `1/[2(n_a+lambda)]`. Retraction maps each state perturbation onto `SE(3)`; apply each process-noise perturbation through the **nonlinear** dynamic equation, producing `s_{k+1}^{(j)}`. Compute their weighted Lie/Karcher mean using consistent left logarithms and reconstruct

$$
P_{k+1|k}=\sum_j w_j^c\,\delta s_{k+1}^{(j)}\delta s_{k+1}^{(j)\top},\quad
\delta s_{k+1}^{(j)}=s_{k+1}^{(j)}\boxminus\bar s_{k+1}.
$$

If a sigma-point weight choice produces a numerically indefinite covariance in a strongly nonlinear regime, report the failure and use a documented sigma scheme with positive covariance weights. Do not silently clip large negative eigenvalues and call them calibrated uncertainty. A stable explicit option is `alpha=1`, `kappa=0`, `beta=2`, giving `lambda=0`, `wm0=0`, `wc0=2`, and all non-central weights positive. Numerical sigma parameters govern the deterministic cubature approximation, **not motion/outlier thresholds**.

### Cross-covariance for backward smoothing

From the **same augmented prediction sigma set**, save

$$
P_{k,k+1}=\sum_j w_j^c (s_k^{(j)}\boxminus \hat s_{k|k})(s_{k+1}^{(j)}\boxminus\hat s_{k+1|k})^\top.
$$

Here `s_k^(j)` is the state component of the augmented sigma point (excluding independent noise), and cross-covariance includes the exact sigma-based nonlinear mapping, not `P F^T` borrowed from EKF.

## B3. Nonlinear matched-point UKF correction without a huge measurement covariance

The outer matcher fixes a set `E_k` of raw RGB-D associations and visibility hypotheses for one conditional update. **Generate fresh state sigma points from the predicted `(s^-,P^-)`** (dimension `n`, count `L=2n+1`), using the same manifold chart and declared positive covariance weights.

Evaluate every conditional matched-point output `h_a(s^(j),p_i)` at each sigma point. If the same match becomes invalid under significant sigma-point variation (occlusion, leaving camera bounds, ordering switch), do not pretend the conditional smooth measurement model remains valid: defer that factor to the outer association/visibility calculation or explicitly model the visibility event probabilistically. A discontinuous hard z-buffer is NOT a smooth continuous-valued UKF output.

For active observation `a`, let `hbar_a=sum_j wm_j h_a(s^(j))`, `e_a=z_a-hbar_a` and sigma-centered columns

$$
Z_a[:,j]=\sqrt{w_j^c}\left(h_a(s^{(j)})-\bar h_a\right)\in\mathbb R^{d_a},\quad
A[:,j]=\sqrt{w_j^c}\left(s^{(j)}\boxminus\bar s\right)\in\mathbb R^n.
$$

Because the covariance weights are chosen nonnegative, the `sqrt(wc)` representation is real. Concatenate `Z_a` vertically into an `m x L` operator `Z` (NEVER materialize a huge persistent m-by-L tensor) and residual `e` into `m`. For the **joint batch** nonlinear unscented approximation:

$$
S=R+ZZ^\top,\quad C_{sz}=AZ^\top,\quad
\Delta s=C_{sz}S^{-1}e,\quad
P^+=P^--C_{sz}S^{-1}C_{sz}^\top.
$$

### Exact Woodbury computation for the UKF's sigma-defined covariance

With positive definite declared observation covariance `R`, accumulate **streamingly**:

$$
\boxed{M=I_L+\sum_a Z_a^\top R_a^{-1}Z_a,\qquad
b=\sum_a Z_a^\top R_a^{-1}e_a.}
$$

These block sums assume the selected raw measurement factors are independent *conditional on the state/map*. For overlapping interpolated pixels or known correlated depth errors use a sparse/shared raw-noise `R^{-1}` operator instead of the block sums. No covariance may be silently treated as diagonal when raw observations are reused.

By Woodbury:

$$
\boxed{\Delta s=A M^{-1}b,\qquad
P^+=P^- - A\big(I_L-M^{-1}\big)A^\top.}
$$

These equations are **algebraically exact for the chosen UKF sigma-point approximation**. They do NOT form the `m x m` innovation matrix `S`, any `m x m` covariance or an `m`-dimensional dense inverse. Solve the small `L x L` matrix `M`; for one group `n=12`, `L=25`, even if hundreds of thousands of matched raw observations are processed. With multiple coupled groups `n=12B` and `L=24B+1`. Use the same conditional observation likelihood as Framework A and check the accepted update against that nonlinear score; damping/line-search may be used for numerical acceptance but **is not an IEKF derivative update**.

The UKF posterior covariance is the sigma-based conditional approximation. A decrease in the objective does not imply this Gaussian covariance is exact for unknown data association or uncertain map points. Shared point-source pose correlations must be represented through retained sparse map-motion information factors or clearly distinguished conditional covariance outputs.

## B4. Unscented backward temporal smoother, not EKF RTS

For each `k=K-1...0`, use the UKF's *saved sigma-point cross-covariance* `P_{k,k+1}` and the already computed predicted covariance `P^-_{k+1}`:

$$
\boxed{G_k^{UKF}=P_{k,k+1}(P^-_{k+1})^{-1}.}
$$

Then smooth in the selected Lie tangent chart:

$$
\hat s_{k|K}=\hat s_{k|k}\boxplus\left[G_k^{UKF}\left(\hat s_{k+1|K}\boxminus\hat s_{k+1|k}\right)\right],
$$

$$
P_{k|K}=P_{k|k}+G_k^{UKF}\big(P_{k+1|K}-P_{k+1|k}\big)(G_k^{UKF})^\top.
$$

Transport covariance when changing the retraction center. This is a sigma-point approximation to nonlinear smoothing, not the analytically linearized RTS update. Do not run an unrelated independent reverse UKF and average poses. All later-observation corrections must act on the SAME shared group trajectories.

## B5. Exactly the shared solved/unsolved outer algorithm

Execute `00_COMMON_SOLVED_UNSOLVED_SPEC.md` verbatim:

1. Fused frame-0 points initialize `SOLVED`. All other raw foreground observations remain `UNSOLVED`.
2. For `k=1..K`, UKF-**predict** the low-dimensional shared group states; transport solved points; sparse RGB-D match; UKF-**correct** conditionally; record unassociated observations.
3. Unscented backward RTS smooth the complete shared trajectory using `P_{k,k+1}` computed from augmented sigma points.
4. Late-to-early discovery from unsolved raw observations: transport under `T[g,k]T[g,s]^-1`, compare raw depth ordering/RGB, reassign to an existing group or propose a new rigid group whose motion is inferred by the **same UKF method**.
5. Fuse accepted cross-frame point matches into one canonical map, without cloning frame/camera depth pixels.
6. Progressive frontier `s=1..K`: resolve unsolved observations against `0..s-1`, revise affected trajectories with UKF prediction/correction and unscented smoothing.
7. Repeat accepted passes under the unchanged common full-sequence score until the outer loop has no further accepted improvement.

There is **no IEKF in this project**. Group identity trajectories are simply valid UKF estimates near `I`, not a different solver.

## B6. Convergence — what is and is not proved

The UKF sigma transform approximates the nonlinear predicted mean/covariance. The resulting update is **not generally a Gauss-Newton descent direction**, and a standard UKF correction need not monotonically decrease a nonlinear MAP objective. Therefore **there is no honest blanket theorem that repeated UKF measurement updates converge to a MAP stationary point**.

A deterministic proposal acceptance rule against the same proper conditional negative log posterior as Framework A can enforce nonincrease of the accepted score. The outer loop accepts every map/group/association edit only after its full comparable sequence score improves, so the sequence of accepted scalar scores is monotone and converges if bounded below. This does **not** prove UKF posterior convergence, global association recovery, or stationary points unless additional descent and regularity hypotheses are independently verified. This limitation must be stated in every output/theory report.

The sigma-point sampling rule is deterministic, not exact integration for arbitrary nonlinear functions. Its algebraic Woodbury computation above is exact **for those sigma-point moments** and does not change the approximation class of the UKF.

## B7. Complexity, verification, outputs

One-group correction uses only `L=25` sigma points and a `25x25` low-rank innovation solve. Augmented 12-state/12-noise prediction uses `49` sigma points. For `B` jointly coupled groups, dimensions scale with `12B` and `24B+1`; state-space size is independent of number of depth pixels. GPU uses shared point transport and sparse matching, then batched sigma-point projections and reductions into `LxL` matrices. Retain `O(N_solved + current_frame_raw + K(12B)^2 + sparse_map_factors)` persistent memory.

**Mandatory tests:** sigma points reproduce a known mean/covariance; zero-noise constant twist agrees with `SE(3)` propagation; UKF vs exact linear Gaussian Kalman update; low-rank Woodbury correction and covariance vs explicitly formed small `S`; matching and visibility tests; correlated-pixel observation noise regression; unscented RTS vs linear-Gaussian RTS on linear fixtures; late-observed geometry back-transports and updates the same shared trajectory; multiple rigid groups split by multi-frame evidence; full forward/backward/frontier run; memory does not scale quadratically in number of measured pixels.

**Deliverable:** Independent stand-alone UKF inner estimator, independent unscented temporal smoother, and the identical solved/unsolved outer SLAM logic. All outputs under `results_ukf/` labeled **UKF** only.


\newpage

# CUDA implementation, shared objective, covariance accounting, and validation

This chapter applies to BOTH independent estimators. The shared CUDA map/projection/matching kernels must be byte-identical across the experiments. Only the 12-dimensional motion solver, its prediction rule and its temporal smoother differ. No model in this chapter calls for normals, mesh surfaces, TSDFs, learned descriptors, or Gaussian rendering.

## D1. GPU representation: compact solved points and indexed observations

Use structure-of-arrays, with immutable 64-bit map point IDs and observation IDs:

| Device buffer | dtype / shape | Meaning |
|:--|:--|:--|
| `xyz` | float32 or float64 `[N,3]` | canonical solved XYZ (not raw pixels) |
| `rgb` | float32 `[N,3]` | point color with declared color-space conversion |
| `group` | int32 `[N]` | estimated rigid-group association |
| `source_id` | int64 `[N]` plus CSR provenance | first observed raw pixel, later supporting observations |
| `map_precision` | compact per-point blocks / sparse factors | conditional map uncertainty and pose-map coupling |
| `solved_owner[c,u]` | int64 `[H,W]` per streamed frame | ID of currently matched canonical point or sentinel |
| `observations` | source RGB, metric depth, validity mask | read from disk by one frame/camera batch |

Persistent `UNSOLVED[k]` is an indexed set/bitmap of unexplained masked raw observation IDs; do not copy XYZ and RGB for each observation into the optimization state. Do not allocate all 328 x 3 frames x 640 x 480 observations on the GPU. Pinned CPU streaming, `non_blocking=True`, and frame caches bounded in bytes are allowed. For the RTX 4070 Laptop (8 GiB), memory budgeting must be measured, not guessed.

Camera poses: all transforms explicit; `T_wc` maps camera coordinates into world, and `T_cw=T_wc^-1`. If camera calibration is in OpenGL coordinates, convert exactly once at load time; validate 3D-to-RGBD cross-camera reprojection and round-trip. Intrinsics must be scaled consistently if diagnostic image resizing is performed. Respect per-scene metadata FPS/timestamps, not an assumed 25 Hz for every object.

## D2. Batched transform and projection kernel

For each solved point `i`, its group `g`, timestamp `k`, and camera `c`:

$$
 x_{ik}=R_{gk}p_i+t_{gk},\quad
 X_{ick}=R_{cw,c}x_{ik}+t_{cw,c},\quad
 u_{ick}=\pi_c(X_{ick}),\quad d_{ick}=X_{ick,z}.
$$

CUDA work item: one `(active solved-point, camera)` pair; read one `group[i]` and one small pose block `T[g,k]`. Write projected `u,v,depth`, projected point ID, validity and group ID. Reject negative depth, out-of-image locations and masked-out query regions for *matching*, while preserving the point itself in `SOLVED`.

A sparse projected-depth buffer can be built by pixel-key sorting and segmented depth minimum, or by a deterministic two-stage tile-key reduce. A temporary atomic minimum is acceptable for visibility if deterministic tie handling is implemented separately; test repeatability under identical inputs. A z-buffer does not itself imply every empty pixel is free space: projected canonical points are discrete samples, and uncovered pixels remain `unknown / possibly unsolved`.

Never rasterize continuous surface support unless such support is explicitly represented. In this project, only point projections and their pixel-cell association are allowed. The sparse predicted RGB/depth images must carry validity masks.

## D3. Local association and no arbitrary novelty cutoff

Key each projected candidate by `(camera, pixel/tile)` and build contiguous sorted/CSR candidate offsets. Use a bounded local search region determined by known camera/sample support or a probabilistic projection uncertainty region with a **documented tail-mass guarantee**. Do not use a fixed metric distance as a declaration that a point is new. For each candidate model point `i` and raw observed pixel `a`, compute the conditional negative log likelihood

$$
 \ell_{ia}=\tfrac12(z_a-h_i)^T S_{ia}^{-1}(z_a-h_i)
 +\tfrac12\log\det(2\pi S_{ia})-\log\pi_{ia},
$$

where `S` contains pixel localization, depth, RGB and any associated point/pose prediction uncertainty as appropriate. Include an explicit `unassigned` alternative with declared normalized likelihood `p_U(z_a)` and a coherent model prior. Compute a deterministic local min-cost assignment, with measurement ownership preventing duplicate reuse. Unassigned pixels remain indexed in `UNSOLVED`. This assignment is a sparse *local* optimum, not an exact global association posterior; record search region and coverage. There is no special static branch: identity motion is simply a trajectory hypothesis `T=I`.

Occlusion is assessed by measured RGB-D ray depth: if a predicted point projects behind a reliable nearer measured point, treat it as *possibly hidden* and do not compare its RGB to that nearer point; if the point is in reliable measured free space, evaluate the contradiction under the declared depth/no-return model. Outside the camera and absent depth are uninformative unless the sensor model says otherwise. Never convert a physically occluded point into `UNSOLVED` merely because it is not seen.

## D4. Raw-noise and covariance correctness

The canonical matched-point measurement contains the target pixel coordinates themselves, depth and RGB. If RGB/depth are interpolated instead, several residuals may reuse identical raw pixels; use the induced non-diagonal covariance `R=W R_raw W^T` or retain common raw-pixel factors. Two overlapping bilinear samples are not independent observations. Foreground mask is an observation-domain restriction, not a motion label or a sensor noise variable.

For source-frame inverse warp `p=T_s^{-1}y_s`, and a left pose perturbation `T_s^+=Exp(delta xi^)T_s`, with world-frame observed 3D point `y_s`, the first-order Jacobians are

$$
 J_y=R_s^T,\qquad
 J_{\xi_s}=-R_s^T\begin{bmatrix}I_3&-[y_s]_\times\end{bmatrix}.
$$

Thus, when original source measurement noise and source pose are independent prior to initialization,

$$
 P_{pp}=J_yR_yJ_y^T+J_{\xi_s}P_{\xi_s\xi_s}J_{\xi_s}^T,\quad
 P_{p\xi_s}=J_{\xi_s}P_{\xi_s\xi_s}.
$$

When they are correlated, add the appropriate cross terms. Distinct points initialized using the same uncertain source pose have a **nonzero shared cross-covariance** `P_{p_i p_j}=J_{\xi_i}P_{\xi\xi}J_{\xi_j}^T` plus any shared raw noise. Store this relationship as a sparse latent factor or selected covariance operator; avoid dense point-point covariance. Fusing two points from the same latent source pose as if independent would count the same motion uncertainty repeatedly.

The inner 12x12 `P` is conditional on the represented map and correspondence state. To report a marginal pose covariance, use the retained sparse motion--map factors and, on a gauge-fixed information system,

$$
 P_{TT}^{marg}=(\Lambda_{TT}-\Lambda_{Tp}\Lambda_{pp}^{-1}\Lambda_{pT})^{-1}.
$$

Schur operators and selected marginal-covariance solves may be matrix-free; never instantiate dense `N_points x N_points` matrices. Always label conditional versus marginal uncertainty and specify the local Lie chart. IEKF line-search damping does **not** enter the posterior precision. UKF sigma moments are an approximation to nonlinear posterior moments, even when the Woodbury solve is algebraically exact.

## D5. IEKF CUDA kernels and reductions (A only)

1. GPU transport/projection kernel: evaluate `X,u,d` for currently solved points and all cameras of current frame.
2. GPU sparse association: output compact `(map_index, raw_id, group_id)` plus measurement covariance/index; no dense correspondence matrix.
3. GPU analytical Jacobian kernel: compute `H_a in R^(d x 12)`, residual, and `R_a^-1` action for each accepted match.
4. GPU segmented reduction by group: accumulate `H_a^T R_a^-1 H_a` and `H_a^T R_a^-1 r_a` in float64 into `B` small `12x12` blocks (or sparse block graph for coupled groups); no global point-vector gradient.
5. Small GPU/CPU batched Cholesky/LDLT solve; left-retract poses and re-evaluate the **same conditional objective** for damping acceptance.
6. Forward state prediction stores `(s_pred,P_pred,F)`. RTS backward uses saved matrices; no full measurement storage.

For `N_m` active matches, arithmetic is `O(N_m d*12^2 + B*12^3)` plus projection and matching; memory is `O(N_m)` transient and `O(B*12^2)` solver state, not `O(N_map^2)`.

In CUDA extensions, use C++17/PyTorch custom ops or torch scatter/index_add; for production prefer a fused kernel for the Jacobian/reduction. Deterministic segmented sum must offer reproducible mode to detect derivative/regression changes. Validate float32 projection versus float64 reference; use float64 accumulators for the small matrix, especially across many observations. No use of `torch.autograd.grad` as a substitute for the requested analytical IEKF Jacobian.

## D6. UKF CUDA kernels and low-rank reduction (B only)

For group error dimension `n=12`, use `L=2n+1=25` state sigma points; in the specified augmented process model with `q=12`, prediction uses `2(n+q)+1=49` sigma points. For jointly estimated groups, replace `n` by `12B` and prefer independent per-group UKF updates where the conditional factor graph genuinely separates; retain coupled blocks when cross-body factors exist.

1. Generate state and process-noise sigma points with Cholesky of only the small `(n+q)x(n+q)` covariance. Retract every sigma state in the SAME left Lie chart.
2. Batch transport and project active matched solved points for every sigma point. Use tensor layout `[L,N_active,3]` in GPU-sized chunks; DO NOT allocate `[L,K,C,H,W,3]` over the sequence.
3. For each match `a`, compute nonlinear conditional output `h_a(sigma_j)` and local weighted deviations `Z_a in R^(d x L)`; compute the residual `e_a`.
4. GPU reduce `M <- I_L + sum_a Z_a^T R_a^-1 Z_a` and `b <- sum_a Z_a^T R_a^-1 e_a`. If nonblock noise correlations exist, apply a sparse/shared raw-noise operator instead of independent block accumulations.
5. Solve the `LxL` positive definite system for `M^-1 b`. Form state deviation matrix `A in R^(n x L)` and update `delta=A M^-1 b`, `Pplus=Pminus - A(I-M^-1)A^T`, with Lie retraction and chart transport.
6. Save sigma-based state transition cross covariance from prediction, then use the UNSCENTED smoother. Do not insert analytical `F` into the UKF smoother and call it unscented.

**Derivation of Woodbury implementation.** Writing `Z` as stacked weighted sigma-output deviations and `A` as state deviations, `S=R+ZZ^T`, `P_sz=A Z^T` and `M=I+Z^T R^-1 Z`, we obtain

$$
 A Z^T S^{-1}e=A M^{-1}Z^T R^{-1}e=A M^{-1}b,
$$

$$
 P^+=P^--A Z^TS^{-1}ZA^T
 =P^--A(I-M^{-1})A^T.
$$

No `m x m` innovation matrix is formed. These identities are exact for the sigma-derived covariance approximation; sigma-point approximation itself is not exact nonlinear Bayesian inference. Validate against a small dense `S` reference and use the same `R` and observation factor set as Framework A.

A straightforward sigma-point UKF is **not iteration equivalent to an iterated UKF**. If an iterative UKF is implemented, every iteration must fix the same predicted prior, generate a consistent sigma ensemble for the declared update rule, and avoid repeated likelihood multiplication. Version one can implement a one-shot UKF update followed by exterior correspondence refinement and new temporal sweeps: this is a complete UKF method, not a hybrid.

## D7. Transactional map edits and the global objective

The shared outer loop stores the original raw observations and the current solved-to-raw ownership. Proposed point insertions, reassignments, merges or new groups are **provisional**. Compute the old/new score using exactly the same normalized measurement and unassigned-observation likelihood, motion prior, group/model prior, and all raw observations (or a mathematically certified incremental exact difference that includes every changed factor). Accept only improvements; rollback otherwise. Never compare a score on 10% of pixels to a score on 100% of pixels. Revisit source-observation ownership when merging points so the same source does not contribute multiple independent terms.

An inner filter is allowed to optimize its **conditional** matched-point model; the outer controller must not infer that a local conditional improvement necessarily improves the complete objective after a correspondence change. For a full-scene objective over all pixels, unexplained observations must have an explicit proper `unassigned` likelihood; sparse predicted projections should not claim pixel coverage they do not have. New points cannot win automatically by explaining only the pixel from which they were initialized because the same observation moves from unassigned to matched and complexity/normalization terms remain in the objective.

If the number of possible groups/associations grows, the mathematical space is combinatorial. Neither algorithm solves an unrestricted global assignment problem exactly at full scale; both solve a deterministic sparse-local candidate problem and verify accepted edits against the same declared score. State this honestly. A stationary map with no further accepted edits is not necessarily a globally complete scene.

## D8. Data config, operational and safety constraints

- Input path must be one scene folder or ZIP containing `rgb`, `depth_npy`, masks, per-camera calibration and any timing metadata. Inspect actual naming/casing; do not hardcode `USB`, `blade` or `fridge`. Test path strings with spaces, e.g. external SSD. Input must be read-only.
- For current environment: Python 3.11, PyTorch 2.11+cu128, NumPy 1.26, SciPy 1.17, OpenCV 5, Open3D 0.19. Optional custom CUDA/C++ extensions only with explicit build instructions; a pure Torch GPU backend must exist for portability. Do not bring in diff-gaussian-rasterization.
- Timing: discover frame timestamps, or FPS from selected scene; use `dt[k]=t[k+1]-t[k]`. If unavailable, require override; never infer real seconds from frame numbering. Noise config `P0,Q(dt),R` must be read from previously authorized numerical assumptions and identified as *assumptions* if not sensor-calibrated.
- Outputs: different result roots `results_iekf` and `results_ukf`, same schema; log data and optimizer timings, GPU peak allocated and reserved memory, and every stage's objective.
- Use `torch.cuda.synchronize()` for timing and inspect memory with `torch.cuda.max_memory_allocated()`; avoid comparing only CPU process RSS.
- Never consume dataset ground-truth part IDs, joint values or motion labels for training or matching. Evaluation-only code may load them after estimation.

## D9. Reference experiments and acceptance gates

**Unit mathematics:** finite differences for all IEKF `F,G,H` blocks (left-chart); analytic SE3 exponential/logarithm and adjoint; exact linear Gaussian EKF; IEKF line search fixed prior; raw-pixel covariance reuse; UKF sigma mean/cov; UKF vs linear Kalman; Woodbury vs dense small `S`; unscented RTS vs RTS in linear fixture; source pose/point covariance correlations; gauge invariance under common rigid change of canonical coordinates.

**Data/projector:** check two views on real valid overlapping foreground points, camera axes and sign convention, metric depth; multiple observations of the same scene point must fuse, not duplicate; verify masks are not treated as rigid-body labels; z-buffer makes RGB of hidden points unobservable; a predicted point inside reliable measured free space yields a negative likelihood; off-image/no-return behavior is sensor-model appropriate.

**End-to-end synthetic:** point group with known nonzero 6DoF motion; another group with identity motion discovered as an ordinary group; initially occluded points appear later and are inversely transported into a shared solved point; group splitting only when supported by multiple frames; full forward -> smoother -> backward discovery -> chronological frontier -> another forward pass. Track source IDs and prevent duplicate measurements.

**End-to-end real:** process every timestamp and camera of the selected scene; report original solved point count, after-fusion count, number and provenance of newly explained points, unresolved by frame, group counts and world pose/twist trajectories; point-wise RGB/depth reprojection statistics, total sequence objective history, and peak CUDA memory. Must complete at least one full forward/backward/frontier/second-forward cycle before claiming success. Preserve all logs, including failed proposals.

**Fair IEKF vs UKF comparison:** same loaded frames, fixed initial fused map, identical noise and priors, identical group/correspondence hypotheses, same allowed wall-clock or compute budget in a controlled experiment, separate random seeds only if any unavoidable randomized subroutine; measure registration error, solved coverage, motion fit, calibration diagnostics, memory and runtime. Do not claim one dominates without measured evidence.

## D10. Mathematical claims and convergence scope

For fixed assignments and map, a correctly damped analytical IEKF with smooth conditional residuals admits local sufficient-decrease guarantees as set out in Framework A. UKF sigma-point moment corrections generally do not inherit a Gauss-Newton descent theorem; they may be rejected by the shared score and retried with a new outer association/initialization, without switching to IEKF. Across both frameworks, accept/reject map and trajectory edits against one bounded-below **common** objective; the scalar sequence of accepted objectives is then nonincreasing and convergent. That does not prove global optimization, true uncertainty calibration for nonlinear multimodal data association, or recovery of permanently invisible surfaces.



# Source-to-point mathematical traceability and precedence

The supplied 20 October 2026 *Bidirectional Articulated 3DGS - IEKF theory*, particularly its Sections 13.1--13.6 and Eqs. (41)--(56), **is the source** of the immutable original RGB, independent real-depth validity, exact foreground-masked observation preparation, and the no-artificial-depth rule. Its randomized background identity and static opacity-carving experiment are *not* point-model theorems. The direct point observation equation (M4), mask/occlusion algorithm (M3), projection Jacobian (M7), and sigma-reprojection (M9)--(M10) are *new output-model-specific derivations* in this document.

Interpret all earlier references to ``renderer'', ``visible point'', ``support'', or ``full image'' **only** according to (M1)--(M5). If an earlier passage conflicts with the masked point revision, the masked point revision takes priority. There is no additional visibility configuration required for execution. These mathematical notes specify implementations and acceptance tests; they do not assert that the user's remote workstation pipeline has already completed them.
