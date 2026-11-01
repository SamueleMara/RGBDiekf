# RGBDiekf Phase 1 — Joint initial-geometry and motion inference

**Mathematical specification, repository implementation plan, and validation protocol**  
**Date:** 2026-10-29  
**Status:** Proposed design; not an implementation or evidence that it works.  
**Repository:** `SamueleMara/RGBDiekf` / local `RGBDekf`. The public default branch was inspected; the user's local uncommitted audit and covariance changes may be ahead. Codex **must inspect its actual local HEAD and working tree before editing**.

## 0. Decision record and non-negotiable constraints

1. Phase 1 estimates **independent free point trajectories** for each original frame-0 endpoint point; no physical part groups, rigid coupling, GT joint values, point-to-point 3D matching, surfaces, normals, meshes, or static/dynamic labels. Keep downstream F2/B2/coherence intact. The original frame-T/backward point catalogue is independent and can receive the same algorithm separately after F1 validation.
2. Inputs: calibrated multi-camera RGB, metric depth, foreground binary masks, timestamps, initial point-cloud catalogue with source pixel/camera and uncertainty; 8 cameras and typically 60 frames. A point's RGB-D observations come from its projected pixel in each camera. Visibility is camera-dependent; hidden points receive no RGB-D information **from that camera**. Preserve identity over occlusion.
3. The frame-0 reconstructed geometry is **not exact**; uncertainty of its initial position is a latent *time-independent correction*. Physical motion is a time-dependent trajectory. Do not turn a single initial geometry correction into a permanent speed.
4. **No speed percentile (`v99`), velocity clipping, arbitrary static/moving classification, GT-assisted inference, or arbitrary spatial regularization**. Unknown stochastic parameters must be given defensible physical initial estimates and identified only when data make them identifiable; report uncertainty otherwise.
5. Keep the old implementation callable and bit-reproducible. A new opt-in solver entry point is introduced first; make it the default only after acceptance. Do not alter F2/B2/joint/support as a side effect. Rendered four-pixel pseudomask IEKF remains `off`.
6. Masks remain a **hard admission condition** for each RGB-D sample. Additionally, all transported object points ought to project into the measured foreground silhouette, even when invisible in RGB-D; implement **feasibility diagnostics initially**, then integrate exact/validated feasible trajectory inference separately. Do **not** substitute a rendered-pseudomask objective or reinstate QP/EP. An inconsistent or subpixel-ambiguous mask must be reported as such rather than silently forcing a false point movement.
7. Retain the full posterior covariance, including cross-time coupling where needed for uncertainty in displacement, model parameter learning, and consistency tests. Never rerun on the previous posterior as if it were an independent prior over the same images.
8. Outputs must distinguish `observed`, `prediction-only`, `numerically failed`, `unidentifiable`, and `mask-infeasible` states. No guessed successes.

**Prior experimental evidence to respect:** A′ reproduced A exactly. Adjusting only `R` reduced but did not eliminate static drift; a tighter GT-derived `P_vv,0` suppressed real moving-part trajectories (storage C/D: moving-part error equals 532 mm true displacement); even both changes left 3.7–22.1% synthetic static points beyond 10 cm. Therefore, tuning R and the velocity prior alone cannot be the algorithm.

---

## 1. Exact state, reparameterization, and observability

For a point i, denote its original frame-0 back-projection `x_ref_i` and canonical source RGB `a_ref_i`. Use the **existing 6D state** at each physical time:

\[
 s_{i,t}=\begin{bmatrix}x_{i,t}\\v_{i,t}\end{bmatrix}\in\mathbb R^6.
\]

The **fixed geometry correction** and **physical displacement** are *derived from the smoothed trajectory*:

\[
 b_i=x_{i,0}-x_{\mathrm{ref},i},\qquad d_{i,t}=x_{i,t}-x_{i,0},\qquad d_{i,0}=0.
\]

The identity `x_{i,t}=x_ref_i+b_i+d_{i,t}` is exact, but does **not** provide a new independent source of information. An explicit 9D `[b,d,v]` parameterization is optional and redundant unless the `d_0=0` constraint and covariance correlations are handled exactly. Implement 6D first.

For Gaussian smoothed estimates, the displacement mean and covariance are

\[
 \hat d_t=\hat x_{t|T}-\hat x_{0|T},\quad
 \mathrm{Cov}(d_t)=P_{x_tx_t|T}+P_{x_0x_0|T}-P_{x_tx_0|T}-P_{x_0x_t|T}.
\]

**Must not approximate** `Cov(d_t)` by the sum of marginal covariances. Verify the lag-0/t cross-covariance by dense reference inversion on small synthetic sequences, then use banded solves, triangular recurrences or covariance queries for scalable execution.

**Identifiability:** `b` and an arbitrary persistent per-point/per-camera measurement bias cannot, in general, be distinguished. A frame-0 source measurement plus visible independent camera observations may constrain `b` through distinct geometric Jacobians, but poorly textured planar regions can remain unobservable tangent to the surface. The estimator must report posterior precision eigenvalues / data-only rank (separately from prior regularization). No claim that smoothing alone resolves intrinsic ambiguities.

---

## 2. Motion prior: proper probabilistic dynamics, not speed caps

At interval `dt=t_{k+1}-t_k>0`, set

\[
 F_k=\begin{bmatrix}I&dtI\\0&I\end{bmatrix},\quad
 Q_k=\begin{bmatrix}\frac{dt^3}{3}Q_a&\frac{dt^2}{2}Q_a\\\frac{dt^2}{2}Q_a&dtQ_a\end{bmatrix},
\]

where `Q_a` is a nonnegative 3×3 spectral density (m²/s³) of white acceleration. For a candidate initial `Q_a`, the process is defined and exact for this SDE; when identification is enabled, learn its parameters from temporal data subject to identifiability checks.

Initial distribution:

\[
 s_0\sim\mathcal N\left(\begin{bmatrix}x_{\mathrm{ref}}\\\mu_{v_0}\end{bmatrix},\;
 P_0=\begin{bmatrix}P_{x_0}&P_{xv,0}\\P_{vx,0}&P_{v_0}\end{bmatrix}\right).
\]

- `P_x0` comes from the original 2D pixel/depth backprojection uncertainty, adding calibration uncertainty where available. Do not count the source camera's originating depth pixel as another **independent** frame-0 depth observation.
- `mu_v0`, `P_v0` describe uncertainty **at time 0**; they must not be derived by mapping all subsequent GT velocities into a scene-wide `v99`.
- With no initial-motion information, *the initial velocity prior is not identifiable from the first frame alone*. Do not claim a uniquely correct value. Stage A tests use known synthetic generative priors; later stages can use learned population priors and identifiable likelihood-based hyperparameters, preserving uncertainty.
- If `Q_a=0`, dynamics are deterministic and require care with singular covariances; explicitly handle deterministic constraints rather than silently inserting an unexplained jitter. Positive `Q_a` cases can use normal SPD factorizations.

The constant-velocity model **does not create a static preference**: both `v=0` and constant nonzero `v` have zero acceleration penalty. Real static/moving distinction must come from frame-0 and later observations and a justified initial-motion prior. If the model cannot track a featureless point sliding along a plane, report unobservability; do not invent motion estimates.

---

## 3. Observation model and conditional visibility

Keep existing model in `estimator/projective_terms.py`:

\[
 h_{i,c,t}(x)=\begin{bmatrix}I_{c,t}(\pi_c(x))-a_i\\D_{c,t}(\pi_c(x))-Z_c(x)\end{bmatrix},\quad y=0,
\]

with analytic Jacobian `H_x`, padded by zero velocity columns. Write `r=-h` for Kalman innovation; keep one sign convention throughout the new code and finite-difference tests.

`estimator/projective_sensor.py` samples four measured neighbours with bilinear interpolation only if all four belong to the foreground and have valid depth, and checks same-surface coherence. `estimator/projective_visibility.py` selects a front projected cloud point per pixel; `projective_terms.py` handles observed foreground surface nearer than the predicted point. Preserve per-camera visibility and reason codes. **Predicted cloud visibility and real depth visibility are different tests**; both are necessary but neither provides exact correspondence.

**Start with a fixed, declared observation model and its covariance.** The current `scene/noise_calibration.py` estimates temporal noise from fixed-pixel frame differences (which may include moving surfaces) and pools cross-camera residual squared error into model variance (which may include persistent bias). These must **not** be silently reinterpreted as consistent unbiased independent `R`. Preserve old calibration as a baseline and label it clearly; make the new stage's `R` explicit and validated.

Critical error dependence: The source RGB `a_i` is one noisy observation reused across all cameras and frames. If treated as a fixed known color, the same source appearance error affects every RGB row and induces temporal/cross-camera correlation. Stage A synthetic tests use truly canonical/noiseless appearance or directly modeled source uncertainty. Later stages should consider a per-point latent canonical color `a_i` with an anchor prior and analytically eliminate it per point, or explicitly model its induced low-rank covariance; do not falsely count reused color as independent data. View-dependent illumination may still violate the constant-color assumption.

Frame-0 source depth is already represented through `x_ref,P_x0`; exclude the duplicate source depth likelihood at frame 0 unless implementing a joint raw-pixel generative model that removes that prior duplication. Independent other-camera frame-0 observations are permissible when visibility and covariance assumptions support them.

**Provisional valid-row likelihood:** condition on a consistent visibility assignment for a linearization pass and use Gaussian `N(h;0,R)`; missing/occluded rows contribute no RGB-D likelihood. Changing visibility/gating changes the factorization, so objective comparisons must keep track of the same measurement support rather than using a trivially lower cost after rejecting all observations.

**Hard-mask geometry:** object points should project into `M_{c,t}=1` when within a camera's field of view, irrespective of whether a front surface occludes them. Log strict pixel-footprint feasibility and boundary ambiguity. Do not let a mask row move a point while inside foreground. Defer numerical enforcement until the core temporal estimator passes calibrated RGB-D and temporal tests; then add a separate well-defined constrained inference step, with explicit infeasibility handling. These are physical requirements, not optional heuristic outlier filters.

---

## 4. Mathematically defined Phase-1 posterior

For each point with state trajectory `S_i={s_i0,...,s_iT}`, initial prior `mu0,P0`, dynamics `F_k,Q_k`, and accepted observation factors `O_i,k`, minimize the **negative log posterior** (constants excluded, for fixed covariance and visibility):

\[
 J_i(S_i)=\tfrac12\|s_{i,0}-\mu_{i,0}\|_{P_{i,0}^{-1}}^2
 +\tfrac12\sum_{k=0}^{T-1}\|s_{i,k+1}-F_ks_{i,k}\|_{Q_k^{-1}}^2
 +\tfrac12\sum_{k,c\in O_{i,k}}\|h_{i,c,k}(x_{i,k})\|_{R_{i,c,k}^{-1}}^2.
\]

The initial term and dynamics term are *not extra postprocessing losses*: they are the original stochastic state-space priors. For parameter fitting (Stage C) include appropriate `log|P0|, log|Q|, log|R|` normalization terms; comparing parameter settings on squared residual alone is invalid.

For a local linearization about `S^j`, the normal equations have per-time measurement blocks

\[
 A_{kk}^{obs}=\sum_c H_{c,k}^TR_{c,k}^{-1}H_{c,k},\qquad
 g_k^{obs}=\sum_c H_{c,k}^TR_{c,k}^{-1}h_{c,k},
\]

with `H=[H_x,0]`. Each transition contributes the block-bidiagonal Jacobian `[-F_k,I]`, hence the full 6×6-block **tridiagonal** normal system. Solve `A delta=-gradient` with block Cholesky / block-Thomas or equivalently forward linear-Gaussian Kalman filtering + RTS smoothing. Reference dense solve for `T<=4`, batched sparse solver for production. Do not invert a dense 360×360 trajectory covariance for every point.

In the linear Gaussian case the RTS smoother equals the batch MAP mean; in the nonlinear case, **iterated extended Kalman smoothing (IEKS)** repeatedly relinearizes the full trajectory and is Gauss–Newton. A one-pass RTS smoother does not solve nonlinear drift, invalid sampling or lost visibility.

---

## 5. Ordered Phase 1 execution (specific stages)

### P1.0 — Preserve baseline and freeze contracts

- Record `git rev-parse HEAD`, `git status --short`, source hashes, environment and dataset manifest. Do not commit unrequested changes.
- Preserve old `run_free()` and CLI flags. New opt-in `--free-solver temporal-ieks` (name illustrative; adopt consistent repo conventions) writes a **new output** directory, never overwriting an older run.
- Inputs and outputs indexed by original frame numbers, point ids, camera indices; preserve canonical source ids.
- Explicitly distinguish code paths: old directional free F1/B1 versus proposed batch F1/B1 temporal IEKS; downstream stages remain unchanged.

### P1.1 — Calibrate frame-0 geometry and construct the anchor prior

1. Load frame-0 RGB-D/masks, source metadata and point id/3D coordinates from `scene/endpoint.py`.
2. Compute source pixel/depth Jacobian and `P_x0=J_bproj Sigma_uvD J_bproj^T`; use camera calibration covariance where available, with correct shared-camera correlations or documented conditional calibration assumption.
3. Construct `s0=(x_ref, mu_v0)`, `P0`; initial 3D position remains a *latent variable*, not a hard point coordinate.
4. Evaluate independent other-camera frame-0 measurements without duplicate source evidence. Report frame-0 geometric information and rank, including weakly observed viewing directions. Do not force an arbitrary correction in unobservable directions.
5. Anchor RGB as known in Stage A ideal synthetic tests; use real canonical colors with explicit 'conditional on source RGB' approximation until Stage B calibrates correlated appearance errors.
6. Output `anchor_prior.npz` or memmapped equivalent with `x_ref`, `P_x0`, source, observed independent cameras, `P0` provenance. Treat this as the **original** prior on all outer iterations.

### P1.2 — Inner forward IEKF initialization

For `k=1..T`, in chronological order:

1. Predict `s_minus=F_k s_plus_previous` and `P_minus=F_k P_plus_previous F_k^T+Q_k`.
2. Project *all* currently transported point locations into each camera to determine approximate frontmost visibility; do not let chunk boundaries change occlusion ownership.
3. Independently sample actual RGB-D at each point projection, using foreground and 2×2 validity tests.
4. Build `h,H,R` only for valid, visible rows. A point occluded in camera c gets **no contribution from camera c** but may be updated from other cameras. With no valid rows: exact prediction-only, `P_plus=P_minus`.
5. Do a fixed-prior IEKF linear solve with the existing 6D point information kernel. Evaluate the proposed state with the **true nonlinear interpolated RGB-D function**. Distinguish a statistically unlikely observation from an invalid nonlinear numerical step.
6. Numerical globalization: ensure candidate steps do not masquerade as improvement by jumping pixel cells or invalidating rows. Implement a deterministic line search / trust-region on the *same local probabilistic objective and fixed measurement support*, with a well-defined failure result, not a new image energy, speed limit or extra motion regularizer. At non-smooth pixel-cell borders test actual residual on both sides, not a one-sided Jacobian as if globally valid. Do not accept a worse objective just because it is under a χ² threshold. Then refresh visibility between accepted updates rather than changing cost support mid-comparison.
7. Store filtered and predicted **full** 6×6 states/covariances for every point/frame; `P_{k,k-1}` not needed for basic RTS but lag-one smoothed covariance is needed for EM later. Store the `F,Q`, accepted factor support, and factor diagnostics.
8. For a tracking failure caused by the point escaping the valid observation support, record `measurement_lost`; do not silently convert it to a successful trajectory. Inner pass alone is a *warm start*, not the final answer.

### P1.3 — Outer backward RTS smoother

Given *one internally consistent forward linearization* and its filtered/predicted histories, for `k=T-1...0`:

\[
 G_k=P_{k|k}F_k^T(P_{k+1|k})^{-1},
\]
\[
 \hat s_{k|T}=\hat s_{k|k}+G_k(\hat s_{k+1|T}-\hat s_{k+1|k}),
\]
\[
 P_{k|T}=P_{k|k}+G_k(P_{k+1|T}-P_{k+1|k})G_k^T.
\]

Use Cholesky solves, not explicit inverse. Retain lag-one covariances and block covariance queries for frame-0 displacement. The state at frame 0 is explicitly smoothed, **not pinned to original backprojection**.

### P1.4 — Outer iterated trajectory relinearization (IEKS)

1. Set initial trajectory `S^0` to filtered or RTS-smoothed warm start. Never replace the original `P0` with `P_{0|T}` as a new independent prior.
2. At each outer iteration `j`, project the **entire proposed trajectory** across every frame, recompute physically consistent approximate visibility and RGB-D sample factors, and form the same original posterior objective `J(S;P0,Q,R)`.
3. Linearize `h(x)` at `S^j`, assemble the block-banded 6D system, solve for `delta S` (via IEKS recursions or batched block tridiagonal solve).
4. Check nonlinear objective decrease for candidates `S^j+alpha delta S`. Do not allow objective improvement solely from dropping observations; freeze a factor-support set for line search and separately report support changes. If all support is lost, classify failure and pursue a documented recovery test rather than pretending convergence.
5. Accept a candidate that actually improves the **original** fixed-parameter posterior (within floating point precision); update visibility and sample support at the next iteration, not midway through a step evaluation. If support changes cause oscillation, report it and revisit the observation likelihood (missing-not-at-random/visibility), rather than masking it with arbitrary thresholds.
6. Stop on objective/gradient/stationarity numerical criteria scaled to precision, with an iteration cap for diagnostics. Convergence means no supported improvement, not merely low displacement. Check both `J` and motion GT in tests; a stationary local optimum is not proof of correct tracking.
7. At accepted solution, compute approximate posterior covariance from the final linearized information matrix (Laplace/IEKS covariance); cross-time covariance queries as needed. Report approximate status when visibility or robust mixtures violate Gaussian assumptions.

### P1.5 — Decompose the smoothed trajectory; report observability

Compute `b=x0_smooth-x_ref`, `d_t=x_t_smooth-x0_smooth`, `v_t` and covariance of each. Include *data-only* observability diagnostics (rank / singular values of relevant whitened Jacobian, especially 3D geometric directions and motion parameters) separately from the full regularized posterior. For a weakly constrained parameter, retain a broad posterior and mark not identifiable; do not set it to zero with hidden logic.

### P1.6 — Model/measurement residual audit after trajectory inference

For each accepted row compute **signed** residual `r`, conditional mean by camera and observation geometry (depth, fractional pixel, gradient, source camera, view angle if observable), per-frame residual, temporal autocorrelation and covariance, and source-color dependence. Compare **depth interpolation at correct positions** with bilinear inverse-depth on locally single-surface planar fixtures, using analytically derived gradients and transformed noise. Preserve the original sampler until this alternative passes independent tests.

Distinguish: `E[r]` (bias), `Cov(r-E[r])` (random variation), `Cov(r_t,r_s)` (persistent correlation). A single additive variance matching `E[r²/R]=1` does **not** prove `R` calibrated, especially if normalized median is near zero while mean is one. Do not introduce arbitrary per-point-camera constant bias states: they are confounded with 3D geometry. Introduce only physically structured, identifiable common camera parameters or modeled source appearance error in an optional later module.

### P1.7 — Structured stochastic identification, separately enabled after Stage-A/B verification

Let `theta` parameterize `Q_a`, sensor/model `R` and legitimately identifiable initial-motion distribution parameters. Use approximate EM / marginal likelihood with a **consistent measurement model**. E-step = IEKS posterior, including lag-one covariance `P_{k+1,k|T}`. For each transition `e_k=s_{k+1}-F_k s_k`:

\[
 E[e_ke_k^T]=\hat e_k\hat e_k^T +P_{k+1|T}+F_kP_{k|T}F_k^T-P_{k+1,k|T}F_k^T-F_kP_{k,k+1|T}.
\]

Fit `Q_a` by minimizing the properly normalized expected process negative log likelihood, with `Q_k=Q(dt_k;Q_a)`, **not** a free 6×6 Q for each time. For Gaussian linear measurement `E[h h^T]≈h(mean)h(mean)^T +H P_{x|T}H^T` under first-order linearization; parameter fit includes `log|R|`. For correlated observations, use the correctly specified non-diagonal or augmented low-rank model. Validate held-out likelihood; if `Q` and `R` are confounded report that and do not produce confident fitted values.

`P_{v0}` hyperparameters cannot be learned per point from a single frame. A population/shared model or repeated independent sequences may identify them. Do not infer real-data `P_{v0}` from unavailable GT speed.

### P1.8 — Masks and consistency; staged integration

**Stage 1**: retain exact binary mask/valid depth admission for RGB-D and compute full-frame hard membership diagnostics `M_ct(pi_c(x_it))` for **all transported object points**, not only visible points. Also log at image border and segmentation boundary, not just outside counts. A hidden point remains inside the global object silhouette in normal opaque foreground segmentation; if other objects or incomplete masks violate that premise, explicitly flag the dataset assumption.

**Stage 2 after temporal core passes**: formulate mask membership as a constrained trajectory posterior. Since binary mask membership is non-smooth/non-convex, either use exact feasible-set handling with documented limitations or mathematically justified local constraints and explicit feasibility rechecks. A local signed-distance linearization alone **does not guarantee global feasibility**. No 4-pixel splat, no generic mask-L2 loss, no dense global QP/EP. Test all mask projections for all cameras after every accepted complete trajectory.

### P1.9 — Output and downstream contract

`free_forward_temporal/` (and later `free_backward_temporal/`) should contain:

- `canonical.npz`, source/calibration/config hashes, `frame_times`, solver manifest, objective and acceptance history;
- `filter_mean` `[T,N,6]`, `filter_cov` `[T,N,6,6]`, `pred_mean`, `pred_cov` in chunked memmaps / compressed chunk layout;
- `smooth_mean` and `smooth_cov`, `geometry_correction_b` `[N,3]`, `displacement` `[T,N,3]`, `velocity` `[T,N,3]`, selected displacement covariance queries and marginalized `P_b`;
- per-camera/per-frame visibility reason, valid-row count, residual statistics, no-valid-row spans, mask feasibility diagnostics;
- objective (original posterior), Gaussian log likelihood if evaluated, numerical stability, observability and held-out metrics;
- no unconditional per-point trace files; targeted trace IDs only. No duplicate multigigabyte exports if a memmap already contains data.

The existing coherence extractor may consume smooth trajectories only behind an explicit adapter and only after Phase 1 passes. Do not change downstream behavior automatically.

---

## 6. Surgical repository implementation map

**Existing files to retain/reuse**

- `scene/endpoint.py`: backprojection, source pixel, source covariance; add *non-duplicated source-measurement* contract and explicit anchor provenance without changing baseline behavior.
- `scene/noise_calibration.py`: keep old calibration read-only/baseline; add diagnostics distinguishing signed mean, variance and correlated errors; do not publish a new calibrated `R` solely from second-moment matching at GT.
- `estimator/projective_camera.py`: retain projection and Jacobian; test float64 against independent oracle.
- `estimator/projective_sensor.py`: retain bilinear sampler and exact gradient; add separate prototype inverse-depth sampler behind a test-only option; forbid interpolation across mask/depth discontinuity.
- `estimator/projective_visibility.py`: retain global projected-point frontmost assignment; ensure global full-cloud visibility not per memory chunk and rerun between iterations.
- `estimator/projective_terms.py`: reuse residual/Jacobian/reason codes; add APIs to obtain **ungated** raw candidate rows, structured per-factor statistics and persistent factor IDs for objective support; old gate remains baseline.
- `estimator/point_kalman_cuda.py`: reuse `F`, `Q`, prediction, local IEKF information solve, SPD utilities; add explicit frozen-support nonlinear objective evaluation and numerical globalization in a **new solver path**, not a silent change to existing outputs.
- `estimator/free_projective_run.py`: add opt-in temporal driver + storage hooks; original `run_free` must reproduce unchanged baseline arrays.
- `train.py`: opt-in solver flag; rerun/resume and manifest guard.

**New modules (suggested, adjust to local conventions):**

- `estimator/temporal_history.py`: chunked `[frame,point,6]`, `[frame,point,6,6]` mean/cov storage with provenance; no full GPU-all-points-all-frames allocation.
- `estimator/rts_point_smoother.py`: batched exact 6D RTS, cross-lag covariance and block covariance queries, robust SPD solve.
- `estimator/ieks_point_trajectory.py`: global-trajectory factor assembly, block-tridiagonal 6×6 solver, full-trajectory nonlinear relinearization, objective consistency, support tracking.
- `estimator/temporal_model_identification.py`: optional, initially disabled `Q_a`, initial-state and `R` parameter estimation using approximate posterior cross-moments and normalized likelihood.
- `estimator/temporal_observability.py`: Jacobian rank and uncertainty diagnostics; exact block data precision / Schur complement on synthetic examples.
- `tools/evaluate_temporal_phase1.py`: synthetic GT-only evaluator separate from inference, static error, moving displacement error, b/v accuracy, calibration and visibility.

**API contracts (illustrative)**

```python
@dataclass
class TemporalPointPrior:
    x_ref: Tensor           # [n,3]
    mean0: Tensor           # [n,6]
    cov0: Tensor            # [n,6,6]
    source_camera: Tensor   # [n]
    source_pixel: Tensor    # [n]

@dataclass
class FramePointPosterior:
    predicted_mean: Tensor  # [n,6]
    predicted_cov: Tensor   # [n,6,6]
    filtered_mean: Tensor   # [n,6]
    filtered_cov: Tensor    # [n,6,6]
    accepted_rows: Tensor   # per camera / point; references to raw measurements

@dataclass
class TemporalPhase1Result:
    smooth_mean: Tensor     # chunk [T,n_chunk,6]
    smooth_cov: Tensor      # chunk [T,n_chunk,6,6]
    b_mean: Tensor          # [n_chunk,3]
    d_mean: Tensor          # chunk [T,n_chunk,3]
    diagnostics: dict
```

Do not let an API return only diagonal variances where full 6×6, lag-one or frame-0/time covariance is required. Avoid coupling the trajectory solver to debug pages.

**Algorithm sketch**

```text
prior, images = prepare_original_frame0_prior_and_raw_frames()
params = declared_initial_model_with_provenance()          # no GT, no v99

for outer in model_iterations:                                # stage A: fixed params, no M-step
    warm = forward_IEKF_original_prior(prior, images, params) # at first iteration
    smooth = backward_RTS(warm)                                # mathematical linear Gaussian oracle first
    for ieks_iter in trajectory_iterations:
        support = global_cloud_visibility_and_valid_projective_rows(smooth.mean, images)
        factors = linearize_original_RGBD_and_temporal_factors(smooth.mean, support, prior, params)
        delta = block_tridiagonal_solve(factors)
        candidate = find_true_objective_improving_step(smooth.mean, delta, frozen_support=support)
        if candidate fails: record_failure_not_false_convergence(); break
        smooth = recompute_local_gaussian_posterior(candidate, factors)
        if numerically_converged: break
    b = smooth.mean[0,:3] - prior.x_ref
    d = smooth.mean[:,:,:3] - smooth.mean[0,None,:3]
    report_trajectory_and_covariance_and_observability(smooth, b, d)
    if not enable_parameter_identification: break
    params = fit_identifiable_parameters_using_smoothed_cross_moments(smooth, images, params)
    # next iteration begins with ORIGINAL prior/images; never treat smooth posterior as new independent prior
```

**Algorithmic detail:** The above is conceptual, not directly executable pseudocode. For parallel points, visibility requires complete scene state at a frame while 6D temporal solves can be point-chunked. Recomputing a nonlinear MAP covariance should occur at the candidate's accepted Jacobian rather than reusing the pre-step covariance. If only a subset of rows changes validity, preserve factor identifiers and explicitly assess/support consistency.

**Memory budget:** 300,000 points × 60 frames × 6 × 6 × 8 bytes ≈ 5.18 GB for **one** full covariance history, before predicted/smoothed histories, means, camera rows and cache. Do not allocate all histories on GPU (even a 32-GB GPU can be exhausted by several replicas). Stream per-point blocks on CPU-backed storage, compute visibility per frame over global points, and batch block-tridiagonal trajectory solves on GPU. Profile at 10k, 100k and full 300k points.

---

## 7. Required validation: exact tests and pass/fail evidence

Define validation gates. For each test report `fixture construction`, `noise model`, `ground truth (evaluation only)`, `point count`, `frame count`, `camera setup`, `priors`, `expected invariant`, `observed result`, `reproduction command`, and whether test uses an oracle. **Do not mark a gate passing by decreasing static drift alone**; moving accuracy and uncertainty validity are equally required.

### Gate 0 — Baseline preservation and reproducibility

- G0.1: Run original F1 on existing smoke fixture before/after the change; bitwise equality of prior output arrays, original manifest hashes, accept/status reason codes. `--free-solver temporal-ieks` opt-in only.
- G0.2: Trace-on versus trace-off numerically identical (already expected from the audit); verify the new solver too.
- G0.3: All existing sensor/projective/visibility/point-IEKF tests remain passing; CPU and GPU deterministic where promised.
- G0.4: No use of GT paths or GT speed under inference call graph. Deliberately remove all GT files from test inference input and assert same output.

### Gate 1 — Linear-Gaussian Kalman/smoother mathematics

- G1.1: Random seeded SPD 6×6 initial prior, `Q_a`, `R`, linear 3D position measurements, irregular actual `dt`. Compare batched Kalman+RTS means and all covariance blocks with independent NumPy **dense full-trajectory** Gaussian conditioning at `T=3..8` (float64 residual near machine precision; tolerance derived from condition number).
- G1.2: Compare block-tridiagonal IEKS one-step normal-equation solver to independent dense `J` solve; covariance diagonal and `Cov(x_t,x_0)` cross queries match full inverse.
- G1.3: Missing observation frames: `s_plus=s_minus`, `P_plus=P_minus`, no fake information; future measurements improve earlier smoothed states, with exact dense oracle comparison.
- G1.4: Verify `Q(dt)` has correct dimensions, symmetry, PSD, with irregular intervals; cross covariance sign for reverse direction B1; exact deterministic handling when `Q_a=0` instead of silent singular inversion.
- G1.5: No duplicate frame-0 source depth observation: compare posterior covariance with a single raw factor vs backprojection prior; adding a duplicated source factor must trigger test failure.

### Gate 2 — Correct static equilibrium and fixed geometry recovery

- G2.1: **Noiseless static fully visible**, exact initial point and analytic consistent RGB-D. First-step IEKF gradient exactly zero within floating point precision; entire forward and smoothed trajectory remains at truth and velocity zero; no drift for all 60 frames.
- G2.2: **Noiseless static with known injected constant** `delta` at frame-0 catalogue only (no change to raw images). Full-sequence `b_hat` recovers `-delta`; `d_hat(t)≈0`, `v_hat(t)≈0`. Sweep perturbation direction, within prior-supported size, source-camera location and point incidence.
- G2.3: **Noisy static with known covariance**, repeated independent noise realizations; estimator unbiased in Monte Carlo *under correct generative assumptions*, uncertainty coverage tested using NEES/chi-square **aggregate intervals**, residual time covariance checked; compare `Cov(d)` with oracle.
- G2.4: Symmetric untextured plane tangent degeneracy: **do not require exact tangent motion or geometry recovery**. Data-information rank must correctly indicate unobservability and posterior should remain prior dependent, not collapse to false certainty.
- G2.5: Multiple frame-0 cameras with unequal viewing directions improve identifiable dimensions; ensure provenance and source depth single-counting invariant.

### Gate 3 — Real physical motion is preserved

- G3.1: **Correct frame-0 geometry, known translation** including stationary, constant speed, acceleration and reversal. `b_hat` stays near zero, `d_hat` recovers each trajectory; no static-point labels are available to the estimator. Verify motion along camera Z, image-plane axes and mixed 3D trajectories.
- G3.2: **Known geometry perturbation plus actual motion simultaneously**: inject `delta` into original cloud, leave physical RGB-D sequence unchanged, and confirm BOTH `b_hat≈-delta` and `d_hat≈d_GT` for visible moving points. This is the most important discriminative test: compare against original forward IEKF and an oracle batch solver.
- G3.3: Occlusion then reappearance: hidden point gets **zero RGB-D rows from the occluding camera**, may receive rows from another visible camera, retains identity, and later evidence corrects smoothed earlier states. Exact oracle for a linear measurement dropout sequence; nonlinear camera scene for end-to-end.
- G3.4: Motion too large for one bilinear cell: compare linearized prediction and true sampled residual. A candidate that worsens fixed-support posterior is not accepted. Test large translations across multiple pixels and image-cell boundaries; track lost support explicitly. Successful recovery is not guaranteed from a single local initialization: mark this as an unsolved basin-of-attraction issue if it fails.
- G3.5: Genuine moving support in storage/USB/stapler not suppressed just to reduce static drift. Report GT moving displacement error and percentage within defined spatial tolerances alongside static drift (same evaluation protocol across all methods).

### Gate 4 — Projective measurement-model diagnostics and bias separation

- G4.1: Analytic bilinear `h,H` vs independent central finite differences away from cell boundaries; test exactly on both sides of an integer pixel boundary; no arbitrary selection of the gradient at a nondifferentiable point is used as global evidence.
- G4.2: Analytic planar depth image, known true 3D point at fractional pixels: demonstrate bias of bilinear metric-depth interpolation and exactness (within floating-point precision for single-surface planar case) of bilinear inverse-depth interpolation; compare correct Jacobians and transformed noise. Disable inverse-depth interpolation across mixed surfaces/discontinuities.
- G4.3: Inject known **static geometry error**, known shared **camera depth offset**, and both. With sufficient multiview/point support, recover geometry and shared camera parameter from a constrained physical bias model. Without sufficient rank (per-point-camera free bias), detect gauge ambiguity; do not report a unique solution.
- G4.4: Repeated noisy source RGB used at all frames/cameras: verify induced correlation; naive independent treatment must be detectable by overconfident posterior or failed coverage, while explicit latent-color/low-rank treatment restores calibration in the controlled generator.
- G4.5: Current second-moment `R`: reproduce qualitative heterogeneity (mean standardized residual ~1 but tiny median); demonstrate why fit-on-training residual distribution is not held-out evidence. Evaluate signed mean, median, tails and autocorrelation on disjoint frames/scenes.

### Gate 5 — Structured covariance identification (only after Gates 1–4)

- G5.1: Generate correct linear-Gaussian trajectories under known `P0,Q_a,R`, with sufficient independent tracks and views. Run EM and recover parameters with confidence/likelihood diagnostics, respecting `Q(dt)` and posterior covariance terms. No GT used by estimator.
- G5.2: Compare M-step process residual covariance against independent NumPy Monte Carlo posterior sampling on short sequences; lag-one covariance terms are essential.
- G5.3: Show that setting `Q` solely from finite differences of noisy positions inflates acceleration; corrected smoother expectation addresses it. `P_v0` not reconstructed from GT `v99`.
- G5.4: Held-out noise model: no systematic signed residual trend and plausible NIS/NEES calibration **after accounting for selection and nonlinear approximation**; no inference from mean r² alone. Report residual heterogeneity by camera/depth/gradient.
- G5.5: Identifiability counterexample (e.g. short sequence, plane, unknown camera bias): estimator refuses overconfident `Q,R,P_v0` identification rather than returning spurious precise numbers.

### Gate 6 — Foreground and visibility invariants

- G6.1: Foreground occluder passes between camera and a static background-surface point on same object: occluded point receives **no** RGB-D rows from the occluded camera while visible cameras can update it. No source surface switch.
- G6.2: Across all frames, every point's foreground-mask feasibility is evaluated in all in-view cameras even if RGB-D occluded. No change to a correct in-mask point by mask alone.
- G6.3: Deliberately contradictory masks: detect infeasibility; do not force a fake huge displacement; record pixel/point IDs and diagnostics.
- G6.4: Segmentation boundary and one-pixel off-by-one fixtures; numeric sensitivity and masked interpolation support are explicitly tested.
- G6.5: Pseudomask mode `on` is **not** silently enabled; no rendering-based energy contributes to Phase-1 IEKS.

### Gate 7 — Real datasets, speed, and end-to-end evidence

- G7.1: Run the **same baseline and new solver** on storage, USB, stapler synthetic and laptop/black_box real, plus a fully static fixture. Keep all original observation files, mask geometry, timestamp and point IDs identical; evaluate synthetic GT only in reporting.
- G7.2: Report static median/p90/p99 endpoint error, >10cm share, first-frame injected speed, cumulative velocity vs correction decomposition, moving endpoint GT error, time-to-loss of valid rows, recovered observations after relinearization, and photo/depth likelihood.
- G7.3: Do not let lower static drift conceal moving failure. Especially verify storage drawer: the old C/D variant froze it and missed true 532mm displacement; new solver must improve moving tracking, not merely keep points still.
- G7.4: No GT for laptop/black_box; label constant-depth proxy explicitly and report uncertainty of proxy; no claims of GT motion accuracy. Save selected trajectories/projections and full numeric diagnostics for independent inspection.
- G7.5: Benchmark 10k, 100k and 300k points × 60 frames × 8 cameras on RTX 5090 32GB, GPU peak, CPU RSS, bytes written, runtime per inner step, per outer iteration, per visibility pass, per smoother. No per-frame file explosion or duplicate 5+GB matrices unless requested.
- G7.6: Run exact deterministic regression after code modifications. All existing Phase-1 old-mode numeric outputs match original; no downstream API breakage.

**Acceptance:** A gate is PASS only when its mathematical invariant and implementation oracle pass. End-to-end success additionally requires both static recovery *and* genuine moving-point recovery; any major remaining local-minimum/visibility failure is a documented OPEN issue, not hidden behind covariance tuning.

---

## 8. Recommended implementation commits (surgical, independently reviewable)

1. **C0: contracts, reproducibility, and synthetic generator**. Read-only dataset oracles; no estimator changes.
2. **C1: temporal history and batched exact RTS + dense NumPy oracle** (Gate 1). This can run on old filter output, without claiming it fixes drift.
3. **C2: 6D block-tridiagonal full-trajectory IEKS + exact nonlinear objective and frozen-support step validation** (Gates 2–3). New opt-in solver; original baseline unchanged.
4. **C3: fixed geometry/displacement and cross-time covariance query + observability report** (Gates 2–3).
5. **C4: independent observation-model residual audit, including inverse-depth planar test and source-color correlation diagnosis** (Gate 4). Only correction justified by oracle enters new mode.
6. **C5: likelihood-based identifiable parameter estimation** (Gate 5); only after residual model passes.
7. **C6: full-mask feasibility and, after separately validated, constrained inference** (Gate 6).
8. **C7: all-scene evaluation and downstream integration behind explicit switch** (Gate 7). Never switch default solely on static error reduction.

Each commit should ship changed-file list, exact commands, oracle evidence, regression tests, and a concise limitation report. Do not run overnight/full-scale work before G1/G2/G3 acceptance. Do not rewrite all pipeline stages or introduce rigid grouping in Phase 1.

## 9. Open mathematical issues requiring explicit reporting

- **Observability**: untextured planar movement, hidden point intervals without reappearance, and free camera-point biases are not made identifiable by any smoother.
- **Association/visibility**: projective sampled pixel is not guaranteed to image the physical point if predicted location is already wrong. Outer relinearization may reintroduce valid measurements but may still be trapped in a wrong basin; retain failure status.
- **Innovation covariance**: frame-0 source backprojection and canonical RGB induce correlations; repeated systematic interpolation error is not iid `R`.
- **Parameter learning**: unknown `P_v0`, `Q_a` and `R` can be confounded in short sequences. Maximum-likelihood estimation is meaningful only under correct and sufficiently rich observation models; avoid GT leakage.
- **Hard masks**: nonconvex binary silhouette feasibility with finite image resolution may be inconsistent. Local linear boundary corrections do not guarantee global mask feasibility; explicit verification required.
- **Numerical non-smoothness**: bilinear gradients jump at pixel-cell boundaries. Correct algebra and stable Cholesky alone do not imply globally reliable IEKF or IEKS steps.
- **Scope**: Point motion is per-point without rigid coupling; independent free-point motion on a textureless moving surface may be unobservable and require future group information. This is a physical limit, not a coding bug.

## 10. Research references for mathematical rationale

- Bradley M. Bell, *The Iterated Kalman Smoother as a Gauss–Newton Method*, SIAM J. Optimization 4(3), 1994. DOI: https://doi.org/10.1137/0804035
- R. H. Shumway and D. S. Stoffer, *An Approach to Time Series Smoothing and Forecasting Using the EM Algorithm*, Journal of Time Series Analysis 3(4), 1982. DOI: https://doi.org/10.1111/j.1467-9892.1982.tb00349.x
- T. D. Barfoot, C. H. Tong, S. Särkkä, *Batch Continuous-Time Trajectory Estimation as Exactly Sparse Gaussian Process Regression*, RSS 2014. https://roboticsproceedings.org/rss10/p01.html
