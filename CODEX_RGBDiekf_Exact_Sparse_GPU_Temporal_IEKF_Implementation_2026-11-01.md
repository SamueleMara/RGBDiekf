# CODEX — REPLACE THE DENSE/PYTHON FULL TEMPORAL IEKF WITH AN ACTUAL SPARSE GPU IMPLEMENTATION

**Priority:** P0 — mandatory implementation and validation, not another exploratory report.  
**Repository:** `https://github.com/SamueleMara/RGBDiekf`  
**Verified latest public `main` commit:** `f66d734c8b300c4d69c90e35ccea4fc10615ede3` (2026-11-01, **“new dense implmentation”**).  
**Local checkout:** `/home/samuelemara/Joint3DGS/RGBDekf` (check local `git status`/HEAD; local changes may be newer).  
**Authoritative model:** `docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md` and `RGBDiekf_Codex_Depth_Sorted_Temporal_RGBD_Full_FixedLag_IEKF.md`.  
**Empirical failure:** `reports/temporal_rgbd_batch_failure_analysis_20261101.md` and matching JSON.  
**Scope:** Full fixed-lag *depth-sorted temporal RGB-D* **F1/B1** filtering. Keep `legacy` and `simplified` independent and unchanged except common bug fixes proved necessary.

## 0. NONNEGOTIABLE DELIVERABLE

**Replace** the *current per-factor Python graph, one-at-a-time CUDA linearization, sparse-then-dense local QR/SVD, and per-factor nonlinear reevaluation* with a **real batched sparse CUDA pipeline**. It must evaluate the **same probabilistic model and the same hard constraints**—not a new heuristic or a simplified point filter wearing a `full` label.

A full-cloud, eight-camera, **three-frame** synthetic run must complete **frame 1 and frame 2** with actual estimated means, correct **joint-marginal** covariance, covariance/source-reference correlation retained, validated first-hit ownership, and real temporal observations. Evidence is mandatory. Do **not** launch ten scenes × 60 frames until this gate passes.

**Forbidden:** placeholder class/function/`pass`/`NotImplementedError` in the selected production path; dummy arrays; zero-filled uncomputed frames presented as estimates; `torch.eye` or diagonal-jitter substitutes for missing correlated covariance; ignoring shared raw pixels; dropping graph edges; restricting point count; using source GT; switching to simplified on resource failure; `torch.linalg.inv(P_ii)` to pretend it is a joint marginal; unexplained clamping/gating; CPU-per-factor operations hidden under a CUDA label; just raising time limits. A missing GPU sparse library is a *dependency/implementation blocker to solve or report*, never permission to change the model.

The theoretical model is exact **conditional on its nonlinear local linearization and FEJ fixed-lag approximation**, not globally exact nonlinear Bayesian inference. Use that wording; do not claim more.

## 1. VERIFIED POST-COMMIT FAILURE EVIDENCE

The latest public commit `f66d734` **contains the dense implementation and audit**, not a verified sparse GPU replacement.

- `estimator/temporal_run.py:~143–150`: preallocates full `[K,N,...]` outputs and zero-fills them. Batch left all frames 1–59 unwritten but apparently valid.
- `estimator/temporal_run.py:~36–88`, `initialize_full`: one Python loop per source point; graph adds individual anchor state, clean RGB-D pixel and prior factors.
- `estimator/temporal_run.py:~90–104`, `add_full_rows`: one Python loop per temporal row and its four reference sensor-pixel keys.
- `estimator/temporal_run.py:~229–270`: one Python process-factor insertion per point plus source graph and correlated NIS setup.
- `estimator/temporal_full_graph.py:~37–47`, `UniquePixelRegistry`: dictionary/closure per unique raw pixel.
- `estimator/temporal_full_graph.py:~141–235`: Python variable and factor objects; per-factor callable closures.
- `estimator/temporal_full_graph.py:~280–310`, `linearize`: loops over factor closures, calls tiny `cholesky`, `solve_triangular`, creates per-factor `meshgrid`, tests nonzeros, builds sparse COO **after** doing the expensive work.
- `estimator/temporal_full_graph.py:~310–365`, `components/local_linearizations`: builds Python-component lists and **densifies** each component with `torch.zeros((nrows,ncols))`.
- `estimator/temporal_full_graph.py:~370–465`, `solve_sparse_map`, `cost`, `factor_posterior`: repeated dense QR/SVD and whole-graph factor loops inside line search and posterior computation.
- `estimator/temporal_full_graph.py:~480–594`, `marginalize_expired_exactly`: FEJ marginalization through component dense QR, rebuilding the same full graph repeatedly.
- `estimator/temporal_run.py:~280–330`, `accept`: repeated `graph.state(i,frame)` loops, rerenders previous/current frame visibility and evaluates factor domains on each trial.
- `tools/run_temporal_rgbd_critical.py:~70–109`: loops all full scenes before simplified and continues after child errors; no fail-fast or hard process deadline.

Observed batch: **seven runtime-limit exceptions; one ENOSPC; zero completed post-anchor updates; 130.6 minutes lost; roughly 5.88 GiB written.** For USB ~95,967 points and ~376,538 Python graph variable keys; for Oven ~343,176 points and ~969,472 keys. These are *key counts*, not scalar dimension. Blade exceeded a 900-second nominal deadline, reaching ~2,414 seconds. GPU memory peaks were well below an OOM, so increasing GPU RAM or timeout is not the primary fix.

Read the audited report and actual source in the LOCAL checkout first. Save file hash, `git rev-parse HEAD`, dirty diff, PyTorch/CUDA versions, detected GPU sparse backends and the exact failed batch status. Never assume GitHub commit and current dirty worktree match.

## 2. EXACT PROBABILISTIC MODEL — KEEP IT UNCHANGED

### 2.1 Global point state and dynamics

For each point `i`, frame `k` in processing order (forward or backward):

\[
 s_{i,k}=\begin{bmatrix}x_{i,k}\\v_{i,k}\end{bmatrix}\in\mathbb R^6,\quad
 s_{i,k}-F(\Delta t)s_{i,k-1}=q_{i,k},\quad q_{i,k}\sim\mathcal N(0,Q(\Delta t)).
\]

Use **existing** `estimator/point_kalman_cuda.py::transition`, `process_noise`, `predict`, signed `dt` and supplied calibrated camera geometry. Same original point IDs and all eight cameras. No K moving parts, static labels, normals, surfels, meshes, Gaussian splat states, or velocity clipping.

### 2.2 Single clean latent RGB-D value per raw sensor pixel

For every *unique* observed foreground pixel `a=(camera,frame,flat_pixel)` touched in the lag graph:

\[
 y_a=\ell_a+\eta_a,\qquad \eta_a\sim\mathcal N(0,R_a^{\rm sensor}),\quad
 \ell_a=[R,G,B,D]^\top.
\]

`ell_a` is a **single 4-vector variable shared by ALL references and constraints involving that pixel**. Insert exactly one raw sensor likelihood per key. Do not create multiple independent copies of its noise; do not add the same raw sensor variance again to a full-mode warp factor. Input only mask-positive valid RGB-D; no background input.

Index clean pixels on GPU by an injective `int64` encoding of `(camera,frame,pixel)` or an exact lexicographically sorted tuple. Handle collisions and overflows explicitly. The deterministic GPU `unique/sort` result must match the Python oracle's registry.

### 2.3 Coupled source backprojection — NO independent source XYZ likelihood

Source point `i` originates at source `(c_i,p_i,k0)` with source clean pixel `ell_a` and source subpixel error `epsilon_i=(delta_u,delta_v)`. Initialize using

\[
 x_{i,k0}=\pi_{c_i}^{-1}(p_i+\epsilon_i,\ell_a^{D}),\quad
 \epsilon_i\sim N(0,\Sigma_{\rm pix}),\quad
 v_{i,k0}\sim N(0,\Sigma_{v,0}).
\]

Keep `Cov(x_anchor,ell_source.depth) != 0` induced by this exact parameterization. The anchor chart is 5 DoF `[epsilon_u,epsilon_v,vx,vy,vz]` plus its shared 4D clean pixel. Reproduce the current numerically centered parameterization (`temporal_full_graph.py::state`, `add_coupled_anchor_factors`) and its derivatives **exactly**, not a separate `Pxx` plus source-depth prior. The source-camera provenance remains invariant; it is NOT a permanent camera assignment.

### 2.4 Depth sorted point ownership and camera-local temporal warp

All points project to all cameras. At each pixel the winner is **smallest predicted positive depth** among points projecting to that integer pixel; stable tie by original point ID. Do not use measured depth or RGB to choose winner. Mask/finite-depth validity determines whether the winner produces a *measurement*, not who wins the z-buffer. Nonwinners have zero measurement Jacobian for that camera pixel.

Only for a previously acquired **same-camera** reference at frame `r` within the lag:

\[
 q=p-\pi_c(x_{i,k})+\pi_c(x_{i,r}),\quad
 h_{i,c,k,p}=\sum_{a\in\mathcal N_4(q)}w_a(q)\ell_{c,r,a}
 +e_D[Z_c(x_{i,k})-Z_c(x_{i,r})],
\]

\[
 g_{i,c,k,p}=\ell_{c,k,p}-h_{i,c,k,p}.
\]

`e_D=[0,0,0,1]^T`. Same exact four-neighbor bilinear interpolation and **metric depth**, same valid foreground neighbor test, same depth-continuity rule, same frozen current owner during each local linearization, same re-evaluation of owners/support on accepted nonlinear steps. Direct velocity Jacobian columns are zero. Canonical RGB from another camera is NEVER used as the current temporal prediction.

Position Jacobians (predicted measurement):

\[
 H_k^I=-\nabla I_r(q)J_{\pi,k},\quad
 H_r^I=+\nabla I_r(q)J_{\pi,r},
\]
\[
 H_k^D=J_{Z,k}-\nabla D_r(q)J_{\pi,k},\quad
 H_r^D=-J_{Z,r}+\nabla D_r(q)J_{\pi,r},\quad
 \partial h/\partial\ell_{r,a}=w_aI_4.
\]

The residual `g=ell_current-h` has the negatives of prediction derivatives, plus `+I4` for current pixel. Exact bilinear gradient and finite-difference tests mandatory. Discontinuities at cell boundaries / depth-owner switches remain discrete and cannot be differentiated away.

### 2.5 Full warp likelihood, including the ideal diagnostic

\[
 g_{i,c,k,p}\sim\mathcal N(0,R^{\rm warp}_{i,c,k,p}).
\]

`R_warp` means **additional conditional warp-model discrepancy**, *not* sensor noise and *not* cross-camera mismatch. `--temporal-warp-noise ideal` means `R_warp=0` as an **exact equality constraint** `g=0`, with all raw sensor and state uncertainty still active. `config/measured` covariance must have justified provenance. Do not add epsilon jitter to the equality; do not turn it into a high-weight soft penalty; do not use synthetic GT to calibrate it. Preserve existing declared sensor-noise assumption and label uncalibrated `ideal` as such.

### 2.6 Fixed-lag posterior and FEJ marginalization

The full cost over active states, source chart and unique clean pixels is

\[
 J(z)=\frac12\|a-a_0\|^2_{P_a^{-1}}+
 \frac12\sum\|s_k-Fs_{k-1}\|^2_{Q^{-1}}+
 \frac12\sum_{a\in\text{unique raw pixels}}\|y_a-\ell_a\|^2_{R_a^{-1}}+
 \frac12\sum_{\text{soft warps}}\|g\|^2_{R_{\rm warp}^{-1}}+J_{\mathrm{FEJ}}(z),
\]

subject to **all hard warp equalities** and any remaining FEJ equalities. The source position comes from the chart, not an extra XYZ penalty. `J_FEJ` represents the *complete* Schur boundary prior with its stored linearization point, including cross-point and cross-time blocks.

For locally linearized whitened stochastic residuals `r + A delta` and exact constraints `c + C delta = 0`, solve the full joint constrained least squares. One valid reference formula is

\[
 \delta_0=-C^+c,\quad CN=0,\quad
 \delta=\delta_0+N\arg\min_u\|r+A(\delta_0+Nu)\|_2^2,
\]

with the joint constrained covariance

\[
 P_{\rm joint}=N(N^TA^TAN)^{-1}N^T.
\]

`P_current(i)` is the *6x6 joint marginal block*, not the inverse of a local 6x6 information contribution. Source current state covariance uses its source-chart Jacobians. Covariance for unobserved points must obey the true joint predictive/marginal structure; in the simplified independent model, no-observation means exact prediction only.

When dropping expired variables `v` with retained `u`, preserve the full Gaussian Schur complement

\[
 \Lambda'_{uu}=\Lambda_{uu}-\Lambda_{uv}\Lambda_{vv}^{-1}\Lambda_{vu},\quad
 b'_u=b_u-\Lambda_{uv}\Lambda_{vv}^{-1}b_v,
\]

and the reduced hard constraints exactly. This is **FEJ**: frozen at the marginalization linearization. Do not discard fill edges, force block-diagonal covariance, apply ad hoc eigenvalue truncation, forget constant residual cost, or reinsert a raw pixel already marginalized. The reduction is exact **for that linearized system**.

### 2.7 Simplified mode is an explicit scientific ablation

Preserve `estimator/temporal_simplified_iekf.py`. Same z-buffer, masks, reference lifecycle and temporal warp. It fixes reference states and uses

\[
 R_{\rm eff}=R_{\rm current}+\sum_aw_a^2R_{\rm reference,a}+R_{\rm warp}
\]

with independent per-point `6x6` IEKF, explicitly **missing cross-time and cross-point state/reference covariance**. Full mode may NOT call it as fallback. Legacy remains `--observation-model legacy`.

## 3. MANDATORY NEW GPU DATA REPRESENTATION

Delete **production dependence** on per-factor Python `Factor.evaluate` closures/dictionaries in `TemporalWindowGraph`. Keep that existing implementation renamed `TemporalWindowGraphDenseOracle` (or otherwise separately preserved), runnable only on **small tests** and generating independently comparable dense matrices and posterior blocks.

Implement a true **structure-of-arrays (SoA)** graph in a new module such as `estimator/temporal_sparse_gpu.py` and CUDA/Triton kernels where appropriate:

- Device-resident point states `(N,lag,6)` or equivalent packed active state table, initial anchor charts `(N,5)`, and `int64` original ID mapping. No Python dictionary per point.
- Device-resident unique clean RGB-D pixel table `(U,4)`, sensor means `(U,4)`, covariance/whitening `(U,4,4)` or diagonal where the declared sensor model is diagonal, integer source keys and reuse counts; **one factor per unique key**.
- Packed **typed factor records** for `SOURCE_PRIOR`, `RAW_PIXEL`, `PROCESS`, `TEMPORAL_WARP`, `FEJ_PRIOR`, `FEJ_CONSTRAINT`. For each type use CSR-style offsets and flat variable indices; no Python object/closure per row.
- Factor residual arrays and small Jacobian blocks produced by batched vectorized GPU operations or compiled CUDA/Triton kernels. Use `float64` for the full graph and sparse solve unless validated mixed precision meets independent float64 oracle error limits. Preserve precise units.
- Sparse block Jacobian storage as sorted COO/CSR/BSR, with deterministic segment reductions. Assemble at **block granularity** and avoid `meshgrid` / `torch.arange` per factor. Duplicate entries must be summed exactly; no omitted nonzeros due arbitrary magnitude thresholds.
- Compact source/anchor parameterization with backprojection derivatives; avoid building per-source pixel `torch.tensor`, `.item`, `.tolist` or a closure. A one-time CPU-side metadata decode is allowed; the per-frame hot path remains GPU batched.
- Dynamic reference-cell changes update a compact *index/value structure* and unique registry incrementally; do not reconstruct all Python factors. The first acquisition in camera `c` initializes its reference without injecting a fake temporal measurement.
- Preserve all original eight-camera point ownership, including competing overlapping points. `--pcd-subsample-factor 1` means ALL original points. **Never** use `--focal-controls` for a production acceptance test.

**Explicit required functions** (names may vary but implementations must exist and be called):

1. `build_or_update_unique_pixel_table_gpu(...)` — exact raw `(camera,frame,pixel)` dedup and sensor likelihood, source-pixel reuse, retired-pixel guard.
2. `pack_source_process_warp_factors_gpu(...)` — actual typed factor arrays, no per-point/factor Python loops.
3. `evaluate_factor_residuals_and_blocks_gpu(...)` — all prior, source, process, warp and FEJ factor types; exact ideal constraints kept separate.
4. `apply_J(v)`, `apply_JT(u)` — sparse GPU JVP/VJP from the **same frozen factor and owner cache** as the residual. Implement/test adjoint identity.
5. `assemble_sparse_linearized_problem_gpu(...)` — global sparse stochastic matrix and sparse equality matrix, with indexed frozen linearization data.
6. `solve_constrained_sparse_joint_gpu(...)` — **actual** sparse CUDA solve, with strict residual/constraint/conditioning certification; no dense components except oracle-size components.
7. `extract_joint_selected_marginals_gpu(...)` — mathematically correct current-state blocks plus requested reference/state cross blocks; not inverse of diagonal blocks.
8. `marginalize_lag_boundary_sparse_gpu(...)` — full FEJ Schur information/constraint retention, including fill and constant terms.
9. `evaluate_nonlinear_objective_and_visibility_gpu(...)` — batched trial evaluation, full objective, hard feasibility and support-domain parity.
10. `publish_verified_frame(...)` — persist only computed states, valid covariances, accepted status and authoritative completed-frame metadata atomically.

**If any named operation is still a stub or uses a Python-per-factor path in production, the implementation is NOT DONE.**

## 4. GPU SPARSE SOLVER — MUST SOLVE THE ACTUAL JOINT SYSTEM

The present source's dense QR/SVD inside thousands of components is prohibited on the production path. A sparse Jv alone does not qualify: the **MAP solve, ideal constraints, FEJ lag elimination and joint marginal covariance** must all operate at full-cloud scale.

### 4.1 Preserve exact constraints by sparse elimination, not penalty

For each zero-variance warp, the **current clean pixel** `ell_{c,k,p}` occurs with identity Jacobian `+I4` in `g`. Eliminate/substitute this pivot variable **algebraically in the sparse factor system**, transferring its sensor likelihood and all appearances in future reference factors. This is exact at the current linearization and often removes the majority of explicit ideal-warp equality rows. Do not eliminate a pixel twice when shared, silently overwrite inconsistent equalities, or lose source-depth cross-correlation.

Arbitrary surviving FEJ hard equalities or nonpivotable constraints must be handled by a valid sparse equality-constrained solve (e.g. GPU sparse KKT with justified rank handling or sparse rank-revealing elimination). Every claimed elimination must be checked against the dense constrained nullspace oracle on small graphs, including conflicting constraints, rank deficiency, and repeated pixels. Do NOT construct a huge dense global nullspace `N`.

### 4.2 Sparse direct or certified iterative GPU method — choose based on actual installed capability

First inventory the actual machine: NVIDIA GPU, CUDA, driver, PyTorch, `cuDSS`/cuSOLVER/suitable CUDA sparse direct factorization bindings, Triton and available libraries. **Implement and verify one concrete working GPU sparse joint solver**, not a generic interface that throws for production dimensions.

- **Preferred:** GPU sparse **square-root QR** or sparse direct factorization of an appropriately reduced SPD system (or stable symmetric-indefinite KKT factorization if exact constraints remain). Reuse symbolic sparsity analysis/orderings while the fixed factor topology is unchanged. Do not densify a connected component just because it contains more than a few points.
- If a sparse direct backend is unavailable and a **matrix-free Krylov implementation** is used, ensure the reduction genuinely yields SPD for CG or use the correct symmetric indefinite solver for KKT. Implement a GPU block-sparse preconditioner with fully batched operations, no `.item()` inside the iteration. **Certify** primal residual, equality feasibility, symmetry and energy error against independent dense oracles. Unconverged iteration means explicit failure, never a silently accepted approximate Kalman update.
- For a QR-free normal-equation solver, explicitly audit conditioning: forming `J^T J` squares the condition number. A `float64` sparse solver only qualifies when validated against the dense square-root oracle for difficult ill-conditioned cases and reporting actual residual/condition diagnostics.
- Avoid dense `torch.linalg.qr`, `torch.linalg.svd`, `torch.linalg.inv` on full-cloud components. Retain those only in the small test oracle or a *proven bounded independent* microcomponent where dense solves are faster. The production backend should be GPU sparse regardless of component size; do not claim exactness if it cannot handle newly merged components.
- Reuse factorization/preconditioner across the *same* linearization for MAP, NIS and selected covariance queries. Do not build and solve the same full system repeatedly for diagnostics.

### 4.3 Marginal covariance and Schur are FIRST-CLASS requirements

A fast MAP trajectory alone is **insufficient**. For each completed frame compute the joint-marginal `P_i(t,t)` and requested `Cov(s_t,s_r)`, `Cov(s_t,ell_ref)` from the *full constrained information*.

Use sparse selected inversion or batched solves against the actual joint factorization with basis columns for the required blocks. Do **not** materialize the inverse of the entire system. A repeated sparse solve per queried state can also be too slow; make solves batched/reused and profile this stage separately. If no scalable correct marginal algorithm can be implemented, mark FULL_UNSUPPORTED and **do not** declare full mode complete. Test all required covariance blocks and Schur parity against the dense oracle.

The full FEJ prior must preserve **all** induced fill edges after marginalization. Repeated reference pixels must retain their correlations across the window. Do not convert it to diagonal or independent point priors. A Schur fill explosion is a legitimate resource issue to profile, not a reason to claim equivalent inference after dropping edges.

### 4.4 Correlated predictive innovation is diagnostic, not a per-frame dense startup tax

`temporal_full_graph.py::correlated_warp_nis` currently does large dense component work *before* any update. Replace with sparse-factor-based covariance action using the same factorization; compute aggregate and sampled diagnostics after a successful frame or on selected validated batches, without changing the acceptance likelihood. Do not skip an inference update merely to compute an optional diagnostic that can be deferred. If NIS is used to make an acceptance decision, it ceases to be optional and must be evaluated exactly by a scalable method.

## 5. BATCHED NONLINEAR IEKF AND VISIBILITY

Retain the **fixed-prior** iterated update, hard first-hit depth ownership and objective on a **common measurement domain**. Linearization uses frozen owner and interpolation contributors; acceptance rerenders the actual proposed point states, compares owner/reference/foreground feasibility, and evaluates the full stochastic objective plus exact equalities. No invented gates, soft visibility surrogates, new mask force, or dropping hard pixels to decrease cost.

Replace `solve_sparse_map`'s global 25-trial Python factor `cost()` loop, every-trial pointwise `graph.state()` calls, and historical full-frame rerender with batched GPU cost/Jacobian/visibility passes and **incremental dirty-key checks**. The incremental check must be provably identical to a brute-force all-camera/all-frame domain oracle for small graphs. A trial changing one point may change the pixel winner for another; dirty-set must include all competing pixels/points, and any connected dependencies. Whole true components may be considered together; never split an information-graph component to get acceptance.

Do not call `accept(g)` in a way that *mutates* the sensor-factor set before the common baseline and candidate are compared without a symmetric union of factor keys. A new raw pixel enters once; it must be included in BOTH candidate/baseline domains, with exact sensor likelihoods. Rollback must restore state and reference/owner snapshots correctly. Preserve `last_consistent` semantics without using empty support or vanishing measurements as success.

Record for every iterate: available rows, accepted steps, support changes, first-hit owner switches, current/reference pixel registration, stochastic cost, equality feasibility, linear solve certificate, posterior NEES if GT evaluated post-hoc, and precise rejection reason. A successful linear solve with `accepted_updates=0` is **not** tracking success.

## 6. REMOVE OTHER SCALING BOTTLENECKS IDENTIFIED IN COMMIT

- `temporal_run.initialize_full`: vectorize all coupled source anchor setup and source raw pixel registration. Avoid `tqdm(source)`/per-source `raw_values()` calls.
- `temporal_run.add_full_rows`: vectorize `rows.point/camera/reference_frame/reference_pixels/weights` into compact factor and sensor tables. Do not `.tolist()` each row.
- `TemporalWindowGraph.state`: vectorized batched state value and anchor Jacobian retrieval. No loop `graph.state(i,fk)` over every point.
- `TemporalWindowGraph.cost`: fuse residuals and whitened costs by factor type on GPU; don't reevaluate Python closures.
- `temporal_run` after accepted frame: eliminate `for a in range(len(rows))` recomputing each temporal prediction/reference clean value. Use batched gather and `temporal_prediction` on all rows.
- For exact sparse owners: reuse `estimator/temporal_zbuffer.py` deterministic depth sort, but avoid returning scalar GPU values to CPU for each competition. Dense full-image owners only for selected diagnostic frames; sparse winners remain authoritative for inference.
- For component connectivity: use GPU sort/union-find/segmented connectivity (or batched CPU symbolic phase only when PROFILING proves it cheaper and not per-factor) over the true sparse variable incidence including FEJ fill. Component detection does not split a factor. Detect real maximum size rather than current preflight's *lower bound* based only on six scalars per point.
- CUDA synchronizations: use GPU events for timings, aggregate reductions before copying a few scalars to CPU. Instrument `torch.cuda.synchronize` or the profiler to prove there is **no per-row/per-point sync** in the hot path.
- Ensure full mode computes both directions using the signed physical timestamps and processing-order lag, not an assumed chronological previous frame.

## 7. OUTPUT / DISK / BATCH SAFETY — FIX BEFORE RUNNING ANYTHING LARGE

### 7.1 No fake trajectories

The present `temporal_run.py` fills all 60-frame `.npy` arrays with zeros. A failure after frame 0 leaves zero tails that viewers and batch summaries can mistake for motion estimates. Replace with:

- Atomic `completed_frames.json` / `frame_valid.npy` / stage status, updated **only after a frame's posterior mean, covariance, accepted measurements and debug record have been committed**. One authoritative `last_completed_frame` and direction-aware chronological index.
- Either (A) write per-frame chunks and consolidate into legacy `[K,N,...]` layout **only after complete success**, or (B) retain memmaps but initialize uncomputed states as **NaN and an explicit false-validity bitmap**. The viewer and GT evaluator **must refuse** invalid frames, not display zero or NaN as motion. On failed output, partial frames are labeled PARTIAL; missing frames remain invalid.
- Verify stored bytes against NumPy headers and checksum frame chunks. ENOSPC or truncation must never leave `status=COMPLETE` or silently create a viewer.

### 7.2 Storage and deadlines

- Precompute worst-case disk requirements for requested point count, frames and enabled arrays; check available bytes + configurable **operational** headroom; use streaming output and bounded debug export. Do not allocate every huge optional output upfront. Do not delete anything from the user's SSD or move data without consent.
- Save minimal failure metadata in an independent small reserved status location or atomic temporary file. Handle ENOSPC even during report creation. Finalize `batch_summary.json` in `finally` with every requested scene/mode status.
- Wrap child processes in a **hard OS-level wall-clock timeout** with process-group termination (SIGTERM, then SIGKILL after bounded grace), not only cooperative `check_budget()`. Preserve exit code, signal, final frame validity, log and GPU memory. No new processes survive deadline.
- Batch **fail-fast by default** after the first `FULL_RUNTIME_LIMIT`, `FULL_RESOURCE_LIMIT`, solver certificate failure, disk failure or incomplete filtering. `--continue-on-error` may opt in explicitly with a warning. No default loop running 10 identical failures. Run simplified independently rather than waiting until all full scenes finish.
- Never treat a correct scene discovery, correct metadata, a geometry preflight or an HTML frame-0 panel as measured motion estimation.

## 8. TESTS — WRITE FIRST AND RUN ON CPU + CUDA

Extend `tools/test_temporal_rgbd.py` and existing `tests/test_temporal_full_graph.py`, `tests/test_temporal_warp.py`, `tests/test_temporal_motion.py`, `tests/test_temporal_rgbd_pipeline.py`. Add dedicated `tests/test_temporal_sparse_gpu.py`, `tests/test_temporal_sparse_marginalization.py`, `tests/test_temporal_sparse_runtime_safety.py`.

**G0 — Exact model identity, independently checkable**

1. Deterministic source catalogue and original IDs; exact camera/frame/pixel keys; no added or lost point; source backprojection has nonzero correct `Cov(x,clean depth)`.
2. First-depth owner map CPU NumPy vs dense GPU and sparse GPU, including exact depth ties, multiple cameras, collisions, masks and uncovered observed foreground.
3. Same-camera reference initialization and handover A→A+B→B; reverse B1 reference history; missing previous reference means no artificial zero residual.
4. Warp residual, Jacobians wrt both states and all 4 reference latent pixels; finite differences on valid interior cells and explicit flag on nonsmooth cells. No background raw sensor variables.
5. GPU factor record counts and unique sensor registry exactly equal to dense Python oracle. Reused raw reference pixel contributes ONE independent sensor factor; no source-depth information counted twice.

**G1 — Sparse operator and normal-equation identity**

6. For small independent/sharing/cyclic graphs, assert GPU `r`, sparse `J`, `C`, whitened blocks, `Jv`, `J^Tz` and full normal-gradient match dense CPU float64 oracle to documented floating-point tolerances. In particular `dot(Jv,z) == dot(v,J^Tz)`.
7. Covariance whitening matches the **4x4 full SPD** declared warp covariance, not only diagonal RGB/depth cases. `ideal` has separate exact `C` and never Cholesky of zero.
8. Solve same constrained and unconstrained models via CPU dense QR/SVD vs GPU sparse; compare MAP, objective, KKT primal/dual residual, equality residual, rank detection, posterior 6x6 marginals and cross blocks. Include near-singular but identifiable cases and genuinely underdetermined cases that must fail visibly.
9. Verify sparse constraint elimination `+I4` for current latent pixel retains shared-reference correlations and correct raw likelihood, including multiple consumers and repeated references. Conflicting equalities must be identified, not overwritten.

**G2 — True FEJ lag information**

10. Slide lag after 2–3 frames; compare current/reference state mean and covariance, source state–pixel covariance, point–point cross covariance and complete FEJ information against brute-force dense Schur on a small graph. Check exact residual constant and no marginalized pixel re-entry.
11. Test FEJ prior with **fill coupling points not directly linked by a current warp**. Full sparse GPU must keep that induced edge; it must not publish independent 6x6 posteriors.
12. Device gather/selected inversion of all current blocks equals slices of the independent **joint inverse**, never local information inverse. Test large and small connected components and full vs simplified expected differences.

**G3 — Nonlinear IEKF and mask/visibility semantics**

13. At every accepted iteration confirm stochastic full objective nonincrease (under the documented fixed common-domain policy), true equality feasibility, valid hard foreground support, no lost factors, no newly invented pixels without likelihoods.
14. Global brute-force rerender/acceptance vs incremental GPU dirty-owner/cache implementation gives **identical decisions**, even when an updated point steals another's pixel, reference cell changes, and a component's FEJ fill makes several points correlated.
15. Test no accepted step => unchanged posterior mean and covariance appropriate to the selected model; test an independently prediction-only point; no fake covariance shrinkage or false `observed` flags.
16. Test `--mode full` NEVER imports/calls simplified update and cannot silently choose dense oracle or identity covariance for large graphs.

**G4 — Full-cloud production execution and evidence (MANDATORY)**

17. 3-frame **full point cloud** with 8 cameras on `USB_100109` and `blade_103706`; `--pcd-subsample-factor 1`, **NO FOCAL CONTROLS**. Must save actual frame 1 and frame 2 posterior means and valid SPD/PSD appropriately constrained covariance, accepted update counts, real RGB-D residuals, live frame-1/2 HTML panels, and a clean final status.
18. Run numerical parity on selected actual components from those *real full-cloud runs* against an independent dense oracle (only selected small components), plus real sparse graph/covariance consistency witnesses for larger components. The oracle may load recorded linearizations; it must not generate the production estimate.
19. Run a **scaling experiment** on naturally nested original-point subsets *only for performance measurement*, not accuracy acceptance: record 1k, 10k, 50k, all points; report assembly, sparse factorization, solve, posterior selected inverses, Schur, nonlinear acceptance, viewer, peak GPU/CPU RAM, `nnz`, largest true component and fill. Do not claim asymptotically linear solve when actual fill-in is superlinear.
20. At least one 11-frame full-cloud pilot after G4 3-frame success; moving-point GT **post-hoc only**, compare median moving-point errors with legacy. Show source-pixel ownership for Storage 8444/45588, USB 67827/85696/87290 and Blade 27016 when those scenes are used; don't subsample them away.
21. Test hard deadline and fail-fast in a fake child process that sleeps past budget; test ENOSPC simulation, truncated `.npy`, partial `frame_valid` and viewer refusal; batch summary persists and marks failures. No real disk-fill experiment.

Numerical oracle comparisons must be quantitative with **both absolute and relative** errors plus appropriate scale/conditioning; choose tolerances from `float64` solver errors measured against independently computed QR results, not arbitrarily lax limits. No `pytest.skip`, dummy pass, empty fixture, mock CUDA result or test of a function never used by production. Track method version, exact source commit and dirty diff.

## 9. EXECUTION ORDER AND STOP CONDITIONS

1. **Baseline/inspect:** record `git status --short`, `git rev-parse HEAD`, source SHA, source geometry/noise. Read latest `f66d734` implementation and audit. Reproduce one CPU/GPU micrograph exactly and benchmark current source. Freeze oracle outputs.
2. **Implement full SoA source/unique-pixel/process/warp factor builder**. Prove exact parity G0; profile 1k/10k/50k construction. Do not add placeholders.
3. **Implement batched residual/Jacobian and sparse `J/J^T`**. Prove G1 operator parity; use GPU event profiles and verify no pointwise CPU sync.
4. **Implement actual sparse GPU constrained MAP solve** with exact ideal equality elimination and soft warp covariance; validate KKT/QR parity and nonlinear acceptance G1/G3. Only then run a full-cloud frame-1 solve.
5. **Implement exact selected joint marginal covariance and sparse FEJ Schur boundary prior**; validate G2 on real saved components. If this cannot be made scalable, FULL mode is NOT DONE even if MAP solves.
6. **Replace production `TemporalWindowGraph` with the new backend**; keep dense code test-only. Prove `train.py` full mode uses the new methods for **all point counts** and all directions; profile on actual GPU.
7. **Fix output/deadline/runner** safety (Section 7) before any full-cloud run. Confirm full-mode first frame finishes with actual saved valid states and covariance.
8. **Run G4** in increasing scope: 3-frame USB & Blade, then 11-frame Storage & USB (only after success), then 60-frame one scene, finally ten-scene batch only if all gates and disk/runtime estimates pass.
9. **Report failures honestly.** No synthetic motion accuracy claim based on test passes. If the local temporal-translation observation is itself physically invalid on articulated motion, isolate it using true trajectories **offline** and report this scientific limitation separately from solver performance. Do not modify the observation model to hide it without asking for scientific review.

**If any mandatory gate fails, STOP before large experiments.** Prefer finishing fewer but correctly implemented modules rather than fabricating a successful full-scene run or making a large empty change.

## 10. EXACT COMMANDS FOR THE USER — DO NOT RUN UNBOUNDED BATCHES

First, after implementation, from the actual local checkout:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
python -B -m unittest tests.test_temporal_sparse_gpu tests.test_temporal_sparse_marginalization tests.test_temporal_sparse_runtime_safety -v
```

**User-authorized bounded FULL-CLOUD pilot command** (use the existing options after ensuring the runner's fail-fast/timeout fixes are actually implemented):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
bash tools/run_temporal_rgbd_critical.sh \
  --mode full --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 1 \
  --temporal-warp-noise ideal \
  --stop-after free \
  --temporal-runtime-budget-seconds 120
```

`--temporal-warp-noise ideal` is *strictly* an uncalibrated extra-model-error diagnostic: **sensor and reference uncertainty remain active**. The 120-second value is an **operational kill budget**, not an invented noise/association threshold. With the fixed fail-fast runner, failed first scene must stop before Blade. A 120-second time limit is an initial bounded test, **not** proof of feasibility or a tuning knob. Codex should provide a profiler showing the actual measured required runtime if it genuinely exceeds this budget.

Once 3-frame full-cloud passes, the next 11-frame user command is:

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode full,simplified,legacy --direction forward --frame-limit 11 \
  --scenes storage_45135,USB_100109,blade_103706,stapler_103111 \
  --pcd-subsample-factor 1 --temporal-warp-noise ideal \
  --stop-after free --temporal-runtime-budget-seconds 600
```

These commands are **proposed after Codex implementation**, not claimed executable in the current failing code. The second command is **not** authorized for an unattended long batch before the gate. Codex must print exact reproduction commands and leave large production execution to the user unless separately authorized.

## 11. REQUIRED CODEX FINAL REPORT — NOT A STATUS-ONLY ANSWER

Create `reports/temporal_sparse_gpu_exact_implementation.md` and `reports/temporal_sparse_gpu_validation.json` with:

- Local git HEAD/commit and dirty diff, detected GPU(s)/CUDA/backend version, actual sparse API installed or implemented, profiler traces and kernel names; report whether CUDA sparse compute and factorization **actually ran**.
- File/function/line map of every removed Python factor hot loop and replacement GPU kernel/operator, plus a test proving those new functions are actually called in production.
- Factor-type counts, raw pixel dedup count, graph variables **and scalar dimension**, constraint rank, `nnz(J)`, `nnz(C)`, component distribution, fill-in, selected marginal strategy and FEJ boundary size.
- CPU dense oracle versus CUDA sparse errors on full objective, residual/Jacobian, MAP, constrained feasibility, current covariance, source-state/reference-pixel cross-covariance, lag Schur, and adjoint identity.
- Exact measured per-stage timings, GPU peak, host RSS, remaining SSD bytes and output sizes, with actual completed non-anchor frames and accepted update counts.
- Forward/backward correctness, post-hoc moving and static point errors, named-point traces, and actual temporal HTML frames; say `NOT VALIDATED` for anything not run.
- Real CLI commands/return codes, first failure and stopping reason. Explicitly distinguish **compile/test pass**, **numerical correctness**, **full-cloud inference executed**, **motion accuracy**, and **scientifically calibrated noise**.
- **Final mandatory verdict** `FULL_SPARSE_GPU_ACCEPTED` only if G0–G4 all pass; otherwise `FULL_SPARSE_GPU_NOT_ACCEPTED` with exact blocker. No optimistic wording.

Do not commit or push without user authorization. Preserve original legacy behavior and all original results. Do not clean up the failed 5.88 GiB batch unless the user explicitly approves deletion.
