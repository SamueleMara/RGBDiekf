# CODEX IMPLEMENTATION CONTRACT — Local-correlated temporal RGB-D point IEKF

**Date:** 2026-11-02  
**Repository:** `https://github.com/SamueleMara/RGBDiekf`  
**Latest public baseline inspected:** `9465cb109f22e4047d7132b3f9d2f22f3f5d633e` (`new dense implmentation 3`)  
**Typical local checkout:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Priority:** Implement a practical, statistically specified **point-local correlated** temporal RGB-D IEKF. Keep legacy, existing `simplified`, and globally correlated `full` modes unchanged as controls.  
**Status at writing:** This is an implementation specification, **not working code or a validation report**. Codex must inspect the actual local HEAD and uncommitted changes first. No long runs before the gates below.

## 0. The decision: what is exact and what is approximated

We want independent **physical point trajectories**, not a spatial point graph, motion-coherence prior, articulation prior, new mesh, normals, surfels, or Gaussians. Keep the existing 6D state, calibrated fixed RGB-D cameras, per-camera first-depth point ownership, foreground masks, and same-camera temporal RGB-D warp.

The original globally correlated `full` estimator jointly infers all point states and shared latent raw image pixels. It attempts to preserve cross-point pixel-noise covariance through full fixed-lag Schur elimination. The current exact Schur fill requires approximately **5.24 TB for USB** and **31.3 GB for Blade** in the documented lag-2 experiment. This motivates a **declared change of covariance approximation**, *not* an attempt to keep claiming globally exact inference.

**New mode name:** `local-correlated` in both `train.py --temporal-mode` and `tools/run_temporal_rgbd_critical.py --mode`. Do not silently alias it to `simplified` or `full`.

**Within each point**, retain exactly, for the chosen linearized local model:

- current 6D state and relevant previous 6D states (or minimal source anchor chart);
- correlations between current and reference states, including position–velocity blocks;
- correlations among raw RGB-D sensor errors **that the same point has already used**, identified by unique `(camera, frame, pixel)` keys;
- correlations between those raw sensor errors and the point's state, including anchor depth;
- all valid cameras of that point in a **single stacked measurement update**, preserving their *within-point* shared-reference covariance;
- posterior covariance across retained local variables and exact local Gaussian marginalization on window expiry.

**Across distinct points**, set posterior cross-point covariance to zero by the declared block-diagonal assumed-density approximation. If points `i` and `j` use the same raw camera pixel, they carry **separate local stochastic copies** of its sensor noise. This is the sole new statistical approximation. Do **not** erase that same pixel's reuse *within one point's own history*. The global z-buffer still makes physical visibility competitive across points, but it does not introduce a probabilistic motion prior or covariance edge.

The approximation is exact only for the *point-local Gaussian linearization* conditional on chosen pixel owners, reference policy, and active supports. It is **not** the same posterior as globally correlated `full`, not globally exact nonlinear Bayesian inference, and not full sequence smoothing. Every run manifest, result comparison, and plot must say this explicitly.

### Nonnegotiable invariants

1. `--observation-model legacy` remains default. `full`, `simplified` remain callable and mathematically unchanged.
2. Original XYZ, colors, source camera/pixel, IDs and source covariance remain intact under point subsampling. Never reassign original point IDs.
3. No hardcoded speed caps, arbitrary innovation gates, motion classes, GT-derived priors, guessed additional warp-noise covariance, or invisible fallbacks.
4. Never use background RGB-D; all four bilinear reference neighbors must satisfy current foreground/validity rules. Depth ordering is among all projected points; a backmost point never gets the winning pixel's measurement.
5. Current measurements must be independent of the source-camera identity: a point can contribute in *any* camera in which it is first-hit and has a valid same-camera history.
6. Process noise `Q(dt)`, transition `F(dt)`, signed backward timestamps, sensor calibration, and raw warp Jacobians are reused from committed code. Do not turn local covariance into diagonal covariance.
7. Every valid image measurement has a well-defined noise provenance; no sensor likelihood can be counted twice **within a point**, even if a raw pixel reappears in several factors.
8. A rejected proposed motion never publishes a fabricated covariance reduction. Incomplete output frames stay NaN and absent from `completed_frames.json`.
9. A globally exact `full` comparison may be made on small problems; it is not a model the new mode claims to reproduce when different points share pixels.

## 1. Exact equations of the new local model

### 1.1 Physical point prediction

For each canonical point `i`:

\[
 s_{i,t}=\begin{bmatrix}x_{i,t}\\v_{i,t}\end{bmatrix}\in\mathbb R^6,\quad
 s_{i,t}^{-}=F(\Delta t)s_{i,t-1}^{+},\quad
 P_{i,t}^{-}=F P_{i,t-1}^{+}F^T+Q(\Delta t).
\]

Use the *existing* `point_kalman_cuda.transition`, `process_noise`, and `predict` policy. No new dynamics.

For a retained local reference state `s_{i,\tau}`, augment the local Gaussian (before the next measurement) with predicted `s_{i,t}` and preserve the temporal cross block:

\[
 \operatorname{Cov}(s_{i,t},s_{i,\tau})=
 F(\Delta t)\operatorname{Cov}(s_{i,t-1},s_{i,\tau}).
\]

Similarly, for any retained local raw-pixel error `\eta_k`, propagate:

\[
 \operatorname{Cov}(s_{i,t},\eta_k)=F\operatorname{Cov}(s_{i,t-1},\eta_k).
\]

All existing cross-covariances among retained local reference states and pixel errors remain as they were. Append independent process noise **once**.

### 1.2 Raw image and shared-within-point sensor uncertainty

For each distinct raw image key `k=(c,f,p)` that **this point** uses, the sensor gives an observed 4-vector `y_k=[RGB,D]` and a latent sensor error:

\[
 y_k=\ell_k+\eta_{i,k},\qquad
 \eta_{i,k}\sim\mathcal N(0,R_{sensor,k}).
\]

The physical clean color/depth is `\ell_k=y_k-\eta_{i,k}`. `y_k` is observed data, not an independent extra likelihood to add again. New `\eta_{i,k}` is appended with one calibrated prior `R_sensor,k`; if the key has already appeared in this same point's local window, reuse the existing variable and its posterior mean/cross-covariance. If a previously marginalized key is referenced again, **do not recreate it as independent**. Design reference expiry/retention to make this impossible; otherwise explicitly reject that observation and report why.

**Important:** for two different point indices `i!=j`, `\eta_{i,k}` and `\eta_{j,k}` are independent *by approximation*. Within one point, the same `\eta_{i,k}` is shared in every row that uses raw key `k`.

### 1.3 Temporal RGB-D observation (unchanged physical warp)

For point `i`, camera `c`, current time `t`, camera-specific accepted reference time `\tau`, current winning integer pixel `p`, and projected coordinates:

\[
 q=p-\pi_c(x_{i,t})+\pi_c(x_{i,\tau}),\qquad
 \Delta Z=Z_c(x_{i,t})-Z_c(x_{i,\tau}).
\]

For the four reference neighbors `k_a=(c,\tau,p_a(q))`, with bilinear weights `w_a(q)`:

\[
 f_{i,c,t}(a_i)=
 \sum_{a=1}^{4}w_a(q)\left(y_{k_a}-\eta_{i,k_a}\right)
 +e_4\Delta Z+\eta_{i,k_t},
 \quad e_4=(0,0,0,1)^T,
\]

where `k_t=(c,t,p)` and `a_i` is the stacked local Gaussian variable vector. **Observed equation:**

\[
 \boxed{r_{i,c,t}(a_i)=y_{k_t}-f_{i,c,t}(a_i)=\epsilon_{warp},\quad
 \operatorname{Cov}(\epsilon_{warp})=R_{warp}.}
\]

`R_warp` is only **additional conditional warp-model discrepancy**. Current sensor noise and four reference-pixel noise variables are already in `a_i`; **do not add `R_current + Σ w² R_ref` to `R_warp`** in this mode (that would double count them). Existing `simplified` mode *does* use that independent marginal sum because it lacks these latent variables; it must remain unchanged.

`--temporal-warp-noise ideal`: set **only** `R_warp=0`. This is an explicit uncalibrated diagnostic assumption, not zero raw sensor noise. Other `measured|config` policies remain strictly validated.

### 1.4 Linearization and Jacobians

Use the already-tested `temporal_rgbd_terms.temporal_prediction`. Its `H_t`, `H_ref` are derivatives of **predicted clean RGB-D** with respect to the 6D current/reference point states:

\[
 H_t^{RGB}=-\nabla\ell_{ref}^{RGB}(q)J_{\pi,t},\quad
 H_\tau^{RGB}=+\nabla\ell_{ref}^{RGB}(q)J_{\pi,\tau},
\]
\[
 H_t^D=J_{Z,t}-\nabla\ell_{ref}^{D}(q)J_{\pi,t},\quad
 H_\tau^D=-J_{Z,\tau}+\nabla\ell_{ref}^{D}(q)J_{\pi,\tau}.
\]

Here `\ell_ref=y_ref-eta_ref` (not the raw image if `eta_ref` has a nonzero posterior estimate). The local predicted-observation Jacobian additionally has:

\[
 H_{\eta_{current}}=+I_4,\qquad H_{\eta_{ref,a}}=-w_a I_4.
\]

Duplicate pixel keys must **sum contributions into the same Jacobian columns**, never create duplicate independent noise columns. Physical state direct velocity columns are zero; velocity is updated through state–position and state–noise covariance. For source anchor chart, use the chain rule through its minimal backprojection parameterization.

For `m_i` valid camera observations, stack all four-channel rows into a single `4m_i`-vector `r_i` with local Jacobian `H_i`. Its current camera noise variables are distinct unless the actual raw key repeats; the reference variables may overlap, and that overlap is represented in `H_i P_i H_i^T`.

### 1.5 Per-point correlated Kalman/IEKF correction

Let `a_i ~ N(mu_i^-, Sigma_i^-)` be the **joint local Gaussian** containing the current state, reference state(s), and retained sensor errors. At iterate `a_i^(j)`, for stacked current observations `y_i` and nonlinear prediction `f_i`, write `r_i^(j)=y_i-f_i(a_i^(j))` and `H_i^(j)=\partial f_i/\partial a_i`. Compute:

\[
 S_i^{(j)}=H_i^{(j)}\Sigma_i^- H_i^{(j)T}+R_{warp,i},
\]
\[
 K_i^{(j)}=\Sigma_i^- H_i^{(j)T}(S_i^{(j)})^{-1},
\]
\[
 \boxed{a_i^{(j+1)}=\mu_i^-+K_i^{(j)}\left[r_i^{(j)}+H_i^{(j)}(a_i^{(j)}-\mu_i^-)\right].}
\]

**FIXED PRIOR:** The same `mu^-` and `Sigma^-` are used for *every* inner IEKF iteration. Do not repeatedly apply covariance updates within the iterations. Reproject/rerender at trial states using the same hard masks and first-hit semantics. Apply the existing common-domain acceptance rules, ensuring consistency of interacting ownership changes; do not invent a new physical gate. Separate pointwise statistical updates from global discrete visibility checks.

After an **accepted** iterate, relinearize there and compute posterior covariance ONCE:

\[
 \boxed{\Sigma_i^+=(I-K_iH_i)\Sigma_i^-(I-K_iH_i)^T
              +K_iR_{warp,i}K_i^T.}
\]

The equivalent information form applies when nonsingular. Use Cholesky/square-root triangular solves, not explicit matrix inversion. Check finite symmetric PSD covariance; test against independent QR and the alternative algebraic covariance formula. Do not inject jitter to conceal loss of rank.

**Crucial benefit of this representation:** even when `R_warp=0`, `S_i` ordinarily remains SPD because the newly introduced **current raw sensor errors have positive calibrated covariance**. No zero-noise KKT construction or artificial tiny variance is needed. If a degenerate row makes `S_i` singular, diagnose its actual rank and use a justified rank-revealing constrained local solve; never silently perturb the model.

An equivalent correlated-noise formula is permissible only if proven equal. If the sensor errors are analytically integrated out and `C_i=Cov(z_i,nu_i)` records correlation of retained point states with reference noise, the correct equations are:

\[
 S_i=H_zP_zH_z^T+H_zC_i+C_i^TH_z^T+R_{\nu},\quad
 K_i=(P_zH_z^T+C_i)S_i^{-1}.
\]

The cross terms `H_z C_i + C_i^T H_z^T` are **mandatory**; using `R_eff=R_current+Σw²R_ref` alone is the old simplified model, not this one. Because the signs of `C` depend on how `nu` is defined, obtain it by eliminating the explicit `eta` variables symbolically/numerically on a small oracle before implementing this optimization. The direct local augmented algorithm above is the reference implementation.

### 1.6 Source anchor and source-depth correlation

Do **not** initialize a point's `x0` and its source pixel's `eta_depth` as independent. The measured source depth is exactly the one used in canonical backprojection. Use a minimal anchor chart as in the existing globally correlated `full` code:

\[
 \theta_{i,0}=(\delta u,\delta v,v_x,v_y,v_z),\quad
 x_{i,0}=R_{cw}^T\left[(y_{source}^D-\eta_{source}^D)
 \begin{bmatrix}(u+\delta u-c_x)/f_x\\(v+\delta v-c_y)/f_y\\1\end{bmatrix}-t_{cw}\right].
\]

The source `eta_source` variable has **one** sensor prior, as above. The pixel-response prior and velocity prior retain the existing approved values. Do not append a second independent canonical `Pxx` likelihood. The nonlinear chart must be differentiated analytically (and finite-difference tested).

**Avoid singular redundant coordinates:** if using the source minimal chart plus noise, do not simultaneously store `x0` as an independent 3D variable linked by a duplicate exact constraint. When the anchor time leaves the local window, retain its Gaussian influence in the current state/reference/noise covariance and drop expired chart coordinates.

### 1.7 Temporal window and exact *local* marginalization — no global Schur matrix

Keep a configurable, fixed processing-order lag `L >= 2` (existing `--temporal-lag-frames`). Store per point only reference states and raw sensor-error keys that remain valid for active measurement history. Track keys by `(point-local owner, camera, original frame, flat pixel)`; within the point, each key has a unique latent error variable. Use processing-order age in B1; chronological timestamps are still signed for dynamics.

A new state is appended by affine Gaussian propagation; new raw-error variables are appended with one independent covariance prior. Apply each accepted new temporal batch once. When an old reference or error is no longer usable and is guaranteed not to reappear, remove it by **Gaussian covariance marginalization**:

\[
 a_i=\begin{bmatrix}a_R\\a_E\end{bmatrix},\quad
 \mu_R^+=\mu_i^+[R],\qquad
 \boxed{\Sigma_R^+=\Sigma_i^+[R,R]}. 
\]

This is the **exact Gaussian marginal** of the point-local posterior; there is no need to form the information Schur complement `Λ_RR - Λ_RE Λ_EE^{-1} Λ_ER` and no global cross-point fill. Marginalizing does not reset the retained covariance or erase cross terms between retained current/reference states and retained sensor errors. The nonlinear posterior remains a local Gaussian approximation after IEKF linearization.

**Reference lifecycle:** `ReferenceTable` picks the latest *accepted valid same-camera* reference. Do not advance references for a point/frame whose measurement or accepted state is invalid. First sighting in a new camera bootstraps without an appearance residual; stale references outside the lag may be reacquired but must not import an independent copy of a previously used, marginalized key. If re-acquisition requires such a key, choose an admissible new reference and report the bootstrap. Validate camera handovers and B1 explicitly.

**Memory bounds:** No `O(N^2)` point covariance or large global KKT. Per-point storage must be bounded by the number of retained local state/pixel variables and lag; empirical and worst-case memory must be measured. A camera-complete worst case can still be expensive, so use *packed variable-length local buffers*, grouping point updates by actual local state/row counts, and process bounded GPU chunks. Do not allocate an oversized all-point padded dense covariance cube. Do not arbitrarily drop local correlations when a point has many references; fail explicitly if a configured resource budget cannot hold an exact local block. Ensure groups/buckets only control numerical batching, **not** physical point coupling.

### 1.8 Visibility is global; covariance is local

The first-depth ownership image is constructed from **all currently selected points**, once per camera per candidate iteration as necessary. A point may be visible in many cameras; its rows use only camera-specific temporal references. Current winners receive current pixels; hidden points receive no direct measurement factor.

Local IEKF state proposals can change first-hit ownership. Jointly validate all proposed positions against globally recomputed ownership and hard masks before acceptance. If one point reveals an out-of-mask point, identify *both* competitors. Do not replace this by independent camera validity or by keeping old owners forever. Do not introduce graph-based physical motion or artificial spatial neighbors. Retain existing conservative fixed-domain checks as the baseline, but report their rejection rates and avoid falsely counting a frozen point as converged.

## 2. Why this is computationally different from the existing full mode

Current `full`: millions of coupled state/pixel unknowns → sparse augmented KKT, expensive selected inverse → potentially enormous exact **global** Schur fill.

New `local-correlated`: each point holds a **small joint posterior** → one stacked multiview innovation `S_i` of at most `4 * (# active cameras)` for each new frame, batched `K_i`, and direct extraction of the retained covariance submatrix when a reference expires. Cross-point pixel dependencies are intentionally omitted. No cuDSS, symbolic fill, Takahashi selected inverse, or FEJ global Schur is needed **in the new mode**. Keep those existing backends unchanged for the original `full` baseline/oracles.

The new local update is **not** merely the existing `update_points_temporal_simplified` with a renamed flag. That code fixes reference states and ignores the correlation between previously assimilated measurement noise and current state. Both of those must be corrected.

**Output:** store `s_plus` and its **6x6 marginal subblock** `P_plus` extracted from `Sigma_i^+` (using chain rule if `s0` is in the minimal source chart). Store or make available the retained reference-state blocks and cross blocks for selected traced IDs. The per-point output covariance is not the globally correlated `full` marginal and must be labeled accordingly.

## 3. Surgical integration map (inspect actual local files first)

| Existing path | Required action |
|---|---|
| `estimator/temporal_rgbd_terms.py` | **Reuse** `temporal_prediction`, `validate_reference_patch`, `sensor_covariance`, `build_temporal_rows`. Add a *small* adapter exposing Jacobian blocks for clean `y_ref - eta_ref` and current `eta_t`; avoid editing existing full/simplified predictions. |
| `estimator/temporal_zbuffer.py` | **Reuse** `first_hit_pixel_owner` and incremental ownership checks; do not implement another renderer. |
| `estimator/temporal_reference.py` | Reuse camera-local accepted references. Add point-local raw-key lifecycle bookkeeping in a separate module, not a second contradictory reference selection policy. |
| `estimator/point_kalman_cuda.py` | Reuse `F`, `Q`, `predict`. Do not change the default filter or legacy acceptance semantics. |
| `estimator/temporal_simplified_iekf.py` | Keep as **unaltered simplified ablation** (including its explicitly approximate covariance). Do not turn it into `local-correlated`. |
| `estimator/temporal_sparse_gpu.py`, `temporal_sparse_solve.py`, `temporal_sparse_marginalization.py` | Keep the globally correlated `full` engine untouched. Only use for independent small-system comparisons. |
| **New** `estimator/temporal_local_covariance.py` | Implement minimal local source chart, variable/key indices, append-and-predict joint Gaussian, raw-error variable reuse, marginalize-by-covariance-submatrix, provenance and shape checks. |
| **New** `estimator/temporal_local_iekf.py` | Implement stacked point-local residual/Jacobians, fixed-prior IEKF, correct Joseph covariance, no-row prediction-only, numerical rank and residual certificates. GPU tensors/batched operations; no per-raw-pixel Python factor closures. |
| **New** `estimator/temporal_local_run.py` | F1/B1 runner using existing catalogues, references, hard masks, z-buffer, visualization/output format and verified-frame journal. Chunk points/bucket dynamic block sizes; respect resource limits. |
| `train.py` | Extend `--temporal-mode` choices to `full|simplified|local-correlated`; dispatch explicitly to new runner. Default legacy remains unchanged; full mode remains full. |
| `tools/run_temporal_rgbd_critical.py` | Extend `--mode` parser/child command. Forward `--pcd-subsample-factor`, `--pcd-subsample-method`, `--pcd-subsample-seed`, lag, noise and frame limit. Keep fail-fast, deadlines and disk preflight. Any legacy full-cloud G4 gate is specific to `full`, not a reason to skip scientific gates for new mode. |
| `utils/temporal_output.py` | Reuse atomic journal/NaN-tail safeguards and hashes. Do not write a zero-filled incomplete trajectory. |
| `debug/temporal_debug.py` | Reuse original RGB-D, reference warps, owner overlay and point cloud. Add local covariance and accepted/rejected diagnostics described below. |
| `tools/test_temporal_rgbd.py` | Register the new test modules; do not delete/regress 75 existing focused tests. |
| `scene/canonical_subsample.py` | **Reuse** existing deterministic `seeded-uniform` selection. No point reordering or per-frame resampling. |

Avoid overwriting global `full` output directories, renaming old outputs, or changing existing JSON meanings without adding a versioned field. Output manifest must record `temporal_mode=local-correlated`, `cross_point_covariance=block_diagonal_assumed_density`, and the precise local history/reference uncertainty policy.

## 4. Implementation order — do not skip stages

### Stage A — independent small-system reference math (CPU float64)

Implement a tiny *single-point, three-frame* reference in NumPy/SciPy with explicit raw-error keys, correlated source-depth anchor, two-time warp, and an exact small QR-based linearized posterior. Include an exact held-out oracle of the covariance using direct constrained projection. Verify equivalence of the explicit local augmented Kalman update to solving the same local linear-Gaussian least-squares system **without** incorrectly treating previous observations as fresh independent likelihoods.

Show that with genuinely independent camera pixels the local formula agrees with corresponding `full` one-point marginal to numerical tolerance. For two points sharing a reference pixel, explicitly show the **difference** from global `full` (the global off-diagonal block is generally nonzero), and ensure the local block-diagonal approximation is reported, not hidden.

### Stage B — batched CUDA local estimator

Implement SoA-style point local buffers and GPU kernels/batched torch operations. Per-point variable dimension differs with the number of active references/keys: group by the actual small shape, avoid all-cloud worst-case padding, process bounded chunks, and retain 64-bit float precision for Gaussian algebra. Build the block-sparse local Jacobian without Python point/camera-pixel closures. Derive local innovation `H Sigma H^T` and cross-covariance `Sigma H^T` with correct repeated-key summation; solve `S` via batched Cholesky or certified rank-revealing local square-root method. Repeat nonlinear IEKF with fixed prior and accepted linearization.

**Do not** approximate the point's temporal covariance to a single `6x6` matrix inside the history, freeze reference states, or replace shared-within-point raw-error keys by independent copies. If shape handling becomes complex, ship a correct chunked/bucketed CUDA baseline before any optional kernel optimization; no empty placeholders.

### Stage C — reference lifecycle + visibility consistency

Verify same-camera references at every iteration, first acquisition in a new camera, occlusion/no-row, mask intersection, foreground/backmost ownership, reappearance, camera handover, consecutive and nonconsecutive valid references, source pixel ID mapping, and positive/negative `dt` in F1/B1. All new local state updates remain subject to the existing global depth-sorted owner validation; print specific reasons for rejected candidates.

### Stage D — local Gaussian expiry

Introduce a three-frame/lag-3 test and a lag-2 test where a reference state's and noise variable's posterior are correlated with current state. Show that slicing the **posterior covariance** gives the same retained mean/covariance as an independently formed local square-root Schur marginal. Check repeated raw pixel use immediately before expiry, key uniqueness, and no reuse after expiry. Do not create global FEJ sparse matrices.

### Stage E — real synthetic motion and performance

Only after A–D pass, execute bounded real inference. First use 3 frames, both directions on a small deterministic subset; next factor 1, 3 frames on USB and Blade. Measure true filtering results and moving-point error. Then 11-frame forward tests of USB, Blade, Storage and Stapler, comparing `legacy`, existing `simplified`, new `local-correlated`, and (only where feasible) globally correlated `full`, always preserving identical point subset, sensor observations and state priors in each controlled comparison.

No 60-frame all-ten-scenes batch before a complete 11-frame validated pilot. Do not run `full` as part of a huge comparison when its G4 remains blocked; label it `NOT_RUN_RESOURCE_LIMIT`, not zero error.

## 5. Mandatory numerical and physical tests

Add `tests/test_temporal_local_covariance.py`, `tests/test_temporal_local_iekf.py`, `tests/test_temporal_local_pipeline.py`, and `tests/test_temporal_local_motion.py`. CPU independent oracles are authoritative for math; CUDA parity must be checked separately.

**Mathematical tests:**

1. **Source-depth correlation:** analytic `Cov(x0,eta_source.depth)` equals independent Monte Carlo/delta-method Jacobian. Check that redundant `x0 + chart` parameterization is not introduced.
2. **Two-time state Jacobians:** finite differences for `H_t`, `H_ref`, clean-value `H_eta_ref`, `H_eta_cur`, RGB and depth; preserve units and zeros in velocity columns.
3. **One-time sensor prior:** identical `(camera,frame,pixel)` key used twice within the *same point* creates one error variable/prior, not two. Show resulting off-diagonal innovation covariance numerically.
4. **Reuse across multiple frames:** reference pixel noise previously assimilated is correlated with current point state; verify `Cov(state,eta)` persists after propagation, IEKF correction, and window expiry. The new answer must differ from `simplified` in the expected controlled example.
5. **Stacked multiview correlation:** two camera rows processed together match independent dense point-local QR; sequential update must match in the linear fixed-support limit if history is correctly maintained.
6. **Ideal warp equality:** `R_warp=0`, raw sensor noise >0, finite SPD local innovation with correct `S=H Sigma H^T`. No noise floor, no global KKT; compare to constrained Gaussian projection.
7. **Nonideal warp:** user-specified SPD `R_warp`, Joseph vs algebraic covariance, no double counted `R_sensor`.
8. **Velocity inference:** zero direct velocity columns but nonzero posterior `v` updates through `P_vx` and cross-time/noise covariance. No-row retains exact prediction and covariance.
9. **Local window slide:** selected posterior covariance submatrix equals exact independent linear-Gaussian elimination, including source and reference noise cross blocks. Test lag 2 and lag 3, both directions.
10. **Reference rebootstrap:** invalid/stale camera reference cannot be silently used with a new independent noise copy. The first clean reappearance in a new camera may bootstrap, never claim an old matching RGB factor.
11. **Cross-point omitted by design:** two-point shared pixel: global dense oracle yields `P_ij != 0`; new mode must report `P_ij=0` (assumption), and the *within-point* posteriors must match the model with independent per-point copies. Do not claim numerical equality to global full.
12. **Linearization and acceptance:** fixed prior through iterations; all accepted rows in the same valid domain; zero additional sensor information on rejected iterates; explicit nonconvergence status.
13. **Rank and conditioning:** textureless/flat-depth regions; underdetermined geometry; positive sensor covariance prevents erroneous singular hard constraints; certificate errors fail clearly without invented damping.

**Renderer and pipeline tests:**

14. CPU/CUDA exact first-hit ownership parity, including duplicate near-depth point competition (Storage 8444 vs 45588).
15. Mask-positive 2x2 interpolation only; no background interpolation; occlusion, revealing competitors and camera transitions.
16. Original point IDs/source identities/trajectory indexing; frame-0 initialization and signed backward prediction.
17. Uniform downsampling factors 1, 2, 4, 8 with fixed seed; identical selection under repeated runs; cameras/image resolution unchanged; factor 1 exactly equals unmodified cloud.
18. NaN tails, disk preflight, crash journal, `require_valid_frame`, deadline fail-fast. A frame without a successful covariance update remains invalid.
19. No code path from `local-correlated` enters `CudaSparseFactor`, `TemporalSoAGraph` globally correlated source or `marginalize_lag_boundary_sparse_gpu`. Make this an explicit test.
20. All existing 75 focused tests continue to pass (unless a documented unrelated test suite result is independently reproduced; never simply delete a failing test).

**Physical falsification tests — synthetic ground truth used OFFLINE ONLY:**

21. At **GT point motion**, compare raw temporal RGB-D warp residual distributions to residuals at estimated states for static vs moving points. Count reference interpolation failures, occlusions, invalid support and owner conflicts separately. If the GT warp itself is inconsistent, report `TEMPORAL_WARP_MODEL_INVALID_OR_BIASED`; do not fix by artificially inflating covariance.
22. Static equilibrium: test measured stationary points (including Storage 8444 and 45588) for erroneous position/velocity drift; quantify initiating camera and row.
23. USB moving tracks 67827/85696/87290 and Blade 27016: per-frame point position error, speed, accepted camera rows, local covariance blocks and reference age.
24. Compare `legacy`, `simplified`, `local-correlated` on the *same* selected points and frames. Report moving/static median, p90/p99 error (m/mm as appropriate), convergence fraction, pixel ownership changes, prediction-only ratio, NIS/NEES **with the correct local approximation qualification**.
25. CPU/GPU performance at 1k, 10k, all points, and uniform factors 1/2/4/8: GPU/host peak, frames/s, camera ownership cost, per-point update cost, reference storage, local covariance bytes and number of rejected/failed estimates. Do not assume downsampling speed is monotonic; show actual measurements.

All numerical tests must have an independent mathematical witness, not compare two functions sharing the same incorrect helper. Ground truth must never be read in the estimator or used to choose sensor model, priors, association or acceptance thresholds.

## 6. Mandatory debugging, visualization, and results

For each processed frame record:

- `frames_completed`, `processed_points`, `observed_points`, `accepted_points`, `no_row`, `rejected`, `nonconverged` (separate counts);
- per-camera winning pixels, mask and depth validity, reference frame/age, 2x2 support, same-camera bootstraps, ownership handovers;
- RGB residual RMS, depth residual RMS **in meters**, prior/accepted objective (same pixel domain), accepted step and nonlinear iteration count;
- local `S` min eigenvalue / rank, covariance marginal SPD, minimum posterior position variance, position–velocity covariance norms, current–reference-state covariance norms, and state–reference-noise cross-covariance norms;
- local raw-error keys per point (min/median/p95/p99/max), repeated key reuse count, source key reuse, expired key count, and any attempted forbidden re-creation;
- GPU allocation/peak, host memory, chunk sizes, bucket occupancy and per-stage wall times; no giant covariance allocation;
- scientific qualifier: `BLOCK_DIAGONAL_ACROSS_POINTS; TEMPORAL_AND_SENSOR_CROSS_CORRELATION_RETAINED_WITHIN_POINT`.

For traced original IDs `storage 8444,45588`, `USB 67827,85696,87290`, `Blade 27016`, save at least:

- prior/posterior `x,v,P_6x6`, full relevant `Cov(s_t,s_ref)`, `Cov(s_t,eta_key)` with key provenance;
- accepted camera, current winning pixel, actual observed raw RGB-D, clean reference prediction, residual and Jacobian;
- list of raw keys, weights, rejected support and ownership competitors;
- reconstruction error against GT saved **only** by offline evaluator.

Reuse actual live HTML RGB/depth/mask/reference-warp/point cloud viewer, and add a small panel showing per-point reference covariance retention and accepted vs prediction-only status. Never call initialization-only pages tracking results. All outputs must be verified by journal hashes and `frame_valid.npy`.

Required report: `reports/temporal_local_correlated_validation.md` plus machine-readable `reports/temporal_local_correlated_validation.json` with source hashes, exact commands, pass/fail gates, error and timing tables, assumption statement, and open blockers.

## 7. Execution gates and exact commands (run only AFTER Codex implements the CLI)

**Environment:**

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
```

**Gate 0 — focused and independent tests (no full-scene inference):**

```bash
python -B -m unittest tests.test_temporal_local_covariance tests.test_temporal_local_iekf tests.test_temporal_local_pipeline tests.test_temporal_local_motion -v
python -B tools/test_temporal_rgbd.py
```

**Gate 1 — minimal actual inference, uniform factor 8, two scenes:**

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode local-correlated --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 8 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 \
  --stop-after free --temporal-runtime-budget-seconds 600
```

**Gate 2 — full point cloud for first three frames, 8 cameras:**

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode local-correlated --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 \
  --stop-after free --temporal-runtime-budget-seconds 900
```

**Gate 3 — verify *local* lag marginalization using frame 3+, start with 11 frames.** Only when Gates 0–2 truly pass:

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode local-correlated --direction forward --frame-limit 11 \
  --scenes USB_100109,blade_103706,storage_45135,stapler_103111 \
  --pcd-subsample-factor 4 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 \
  --stop-after free --temporal-runtime-budget-seconds 1800
```

**Gate 4 — backward camera handover (after forward pass proves local expiry):**

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode local-correlated --direction backward --frame-limit 11 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 4 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 \
  --stop-after free --temporal-runtime-budget-seconds 1800
```

A truncated backward pass uses the endpoint of the **selected prefix** (as current repository semantics), not original frame 59. Report this clearly.

**Gate 5 — full 60-frame, all points, all scenes** is **not authorized** until the smaller pilots produce valid trajectories, measured motion accuracy, memory/time profiles and a justified noise model. The `ideal` option is a diagnostic, not scientific calibration. Codex must provide the final long-run command after completing Gates 0–4; it must not run that batch during implementation.

`--mode local-correlated` is a **new** option. Commands above are intended for the implemented code; they are not valid against commit `9465cb1` as-is. If an option is unsupported, Codex must implement it, not silently replace it with `simplified`.

## 8. Acceptance conditions, stopping rules, and forbidden shortcuts

**Acceptance of the code:**

- All local covariance and physics equations above appear in functioning, callable code; independent CPU tests and CUDA parity pass; no placeholders, `pass`, unused stubs, fake zero residuals, or hardcoded covariance values.
- The new local mode never runs the global dense/sparse component solver, cuDSS, Takahashi inversion, or FEJ global Schur.
- Same-reference-state and raw pixel uncertainty correlations persist **within each point** across time, including local window expiration and camera transitions.
- The first actual frame and second actual frame are published only after accepted state, local covariance validation and journal checks; no zeros for uncomputed frames.
- At least USB and Blade actually process all selected points/all eight cameras for three frames at factor 1; timing and accuracy are measured and honestly reported.
- The mode is labeled a **cross-point covariance approximation** in manifests and comparisons; it is not misreported as globally exact.
- `full`, `simplified`, `legacy` still run as before and focused tests continue to pass.

**Do not claim a tracking success solely because the numerical tests pass.** If the physical warp is inaccurate even at GT, report the failure and stop model changes for review. If local covariance is still too expensive, provide measured counts/bytes and fix data layout before proposing a new probabilistic model. If the GPU runtime or covariance becomes invalid, fail fast and preserve only committed frames.

**Explicitly forbidden:** any new point-neighborhood/spatial graph, clustering or rigid-motion regularization; learned correspondence/feature extractor; normals/meshes/surfels/3D Gaussians; changed RGB-D measurement equations; GT-informed inference; arbitrary gates or velocity thresholds; silent independence of reused pixels **within a point**; default noise floors; fake successful output; increasing timeouts instead of fixing the observed bottleneck; treating uniform subsampling as a proof of full-cloud accuracy.

## 9. Completion handoff to user

When done, Codex must return:

1. Exact Git commit/working-tree state and modified file list.
2. The local state vector used in code and a table from each equation above to its implementing function and an independent test.
3. Actual test totals/failed cases with commands and logs.
4. Frame-0/1/2 publication evidence for USB and Blade, with accepted point counts and posterior covariance validity.
5. Full memory/runtime breakdown and local variable-size distribution.
6. GT-based *post-hoc* accuracy comparison on matched points; if unavailable, mark it unavailable rather than invent it.
7. The exact commands to reproduce successful tests and the next bounded pilots.
8. Remaining limitations, especially neglected **cross-point** covariance and physics of local warp for articulation.

**Primary scientific question:** Does retaining each point's own two-time reference-state and sensor-noise correlations materially improve motion recovery over the old simplified 6x6 IEKF, without requiring the globally coupled covariance model?
