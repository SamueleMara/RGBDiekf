# CUDA implementation, shared objective, covariance accounting, and validation

**MANDATORY FOREGROUND-MASK ADDENDUM:** The binary foreground masks are a hard admissible-measurement and predicted-visible-point feasibility constraint. Read and implement `CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md` before using this document. No background pixel can initialize SOLVED/UNSOLVED, be matched, enter an IEKF/UKF measurement, or be fused into geometry. Occluded solved points remain valid. Identical policy in both pipelines.


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

\[
 x_{ik}=R_{gk}p_i+t_{gk},\quad
 X_{ick}=R_{cw,c}x_{ik}+t_{cw,c},\quad
 u_{ick}=\pi_c(X_{ick}),\quad d_{ick}=X_{ick,z}.
\]

CUDA work item: one `(active solved-point, camera)` pair; read one `group[i]` and one small pose block `T[g,k]`. Write projected `u,v,depth`, projected point ID, validity and group ID. Reject negative depth, out-of-image locations and masked-out query regions for *matching*, while preserving the point itself in `SOLVED`.

A sparse projected-depth buffer can be built by pixel-key sorting and segmented depth minimum, or by a deterministic two-stage tile-key reduce. A temporary atomic minimum is acceptable for visibility if deterministic tie handling is implemented separately; test repeatability under identical inputs. A z-buffer does not itself imply every empty pixel is free space: projected canonical points are discrete samples, and uncovered pixels remain `unknown / possibly unsolved`.

Never rasterize continuous surface support unless such support is explicitly represented. In this project, only point projections and their pixel-cell association are allowed. The sparse predicted RGB/depth images must carry validity masks.

## D3. Local association and no arbitrary novelty cutoff

Key each projected candidate by `(camera, pixel/tile)` and build contiguous sorted/CSR candidate offsets. Use a bounded local search region determined by known camera/sample support or a probabilistic projection uncertainty region with a **documented tail-mass guarantee**. Do not use a fixed metric distance as a declaration that a point is new. For each candidate model point `i` and raw observed pixel `a`, compute the conditional negative log likelihood

\[
 \ell_{ia}=\tfrac12(z_a-h_i)^T S_{ia}^{-1}(z_a-h_i)
 +\tfrac12\log\det(2\pi S_{ia})-\log\pi_{ia},
\]

where `S` contains pixel localization, depth, RGB and any associated point/pose prediction uncertainty as appropriate. Include an explicit `unassigned` alternative with declared normalized likelihood `p_U(z_a)` and a coherent model prior. Compute a deterministic local min-cost assignment, with measurement ownership preventing duplicate reuse. Unassigned pixels remain indexed in `UNSOLVED`. This assignment is a sparse *local* optimum, not an exact global association posterior; record search region and coverage. There is no special static branch: identity motion is simply a trajectory hypothesis `T=I`.

Occlusion is assessed by measured RGB-D ray depth: if a predicted point projects behind a reliable nearer measured point, treat it as *possibly hidden* and do not compare its RGB to that nearer point; if the point is in reliable measured free space, evaluate the contradiction under the declared depth/no-return model. Outside the camera and absent depth are uninformative unless the sensor model says otherwise. Never convert a physically occluded point into `UNSOLVED` merely because it is not seen.

## D4. Raw-noise and covariance correctness

The canonical matched-point measurement contains the target pixel coordinates themselves, depth and RGB. If RGB/depth are interpolated instead, several residuals may reuse identical raw pixels; use the induced non-diagonal covariance `R=W R_raw W^T` or retain common raw-pixel factors. Two overlapping bilinear samples are not independent observations. Foreground mask is an observation-domain restriction, not a motion label or a sensor noise variable.

For source-frame inverse warp `p=T_s^{-1}y_s`, and a left pose perturbation `T_s^+=Exp(delta xi^)T_s`, with world-frame observed 3D point `y_s`, the first-order Jacobians are

\[
 J_y=R_s^T,\qquad
 J_{\xi_s}=-R_s^T\begin{bmatrix}I_3&-[y_s]_\times\end{bmatrix}.
\]

Thus, when original source measurement noise and source pose are independent prior to initialization,

\[
 P_{pp}=J_yR_yJ_y^T+J_{\xi_s}P_{\xi_s\xi_s}J_{\xi_s}^T,\quad
 P_{p\xi_s}=J_{\xi_s}P_{\xi_s\xi_s}.
\]

When they are correlated, add the appropriate cross terms. Distinct points initialized using the same uncertain source pose have a **nonzero shared cross-covariance** `P_{p_i p_j}=J_{\xi_i}P_{\xi\xi}J_{\xi_j}^T` plus any shared raw noise. Store this relationship as a sparse latent factor or selected covariance operator; avoid dense point-point covariance. Fusing two points from the same latent source pose as if independent would count the same motion uncertainty repeatedly.

The inner 12x12 `P` is conditional on the represented map and correspondence state. To report a marginal pose covariance, use the retained sparse motion--map factors and, on a gauge-fixed information system,

\[
 P_{TT}^{marg}=(\Lambda_{TT}-\Lambda_{Tp}\Lambda_{pp}^{-1}\Lambda_{pT})^{-1}.
\]

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

\[
 A Z^T S^{-1}e=A M^{-1}Z^T R^{-1}e=A M^{-1}b,
\]

\[
 P^+=P^--A Z^TS^{-1}ZA^T
 =P^--A(I-M^{-1})A^T.
\]

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
