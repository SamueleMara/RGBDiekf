# CODEX — SURGICAL COMPLETION OF THE EXACT TEMPORAL RGB-D POINT IEKF

**Date:** 2026-11-01  
**Repository:** `https://github.com/SamueleMara/RGBDiekf`  
**Verified public HEAD:** `f77492dc0c8fe3cf5f28820ba5fe1f8ed959cb58` (`new dense implmentation 2`), at the time of this specification.  
**Local working directory:** `/home/samuelemara/Joint3DGS/RGBDekf` (verify with `git rev-parse --show-toplevel`).  
**Scope:** complete the *already implemented* point-based depth-sorted temporal RGB-D full fixed-lag IEKF; keep the simplified and legacy paths untouched except for necessary shared CLI/infrastructure wiring.  
**Status of baseline:** `FULL_SPARSE_GPU_NOT_ACCEPTED`; 60 focused tests passed; USB full cloud (95,967 canonical points, 8 cameras) had one accepted nonlinear iteration but the **next cuDSS factorization failed with `INFO=-1`**; no verified frame-1 posterior, Blade not started.  
**Do not equate:** unit-test success, initial frozen-system factorization, one accepted trial correction, or a frame-0 HTML viewer with a completed estimated frame.

## 0. THE NON-NEGOTIABLE MANDATE

**Implement exactly the already agreed theoretical model.** This is **NOT** a request to design a graph-based motion model. The term *factor graph* in the code is only an algebraic representation of the specified joint Gaussian likelihood. **No spatial neighborhood graph, rigid-part graph, learned coherence, mesh, surfel, normal, Gaussian Splatting parameters, graph neural network, appearance feature model, new point-to-point regularization or additional dynamics are permitted.**

- One physical point `i` has one global state `s[i,t] = [x,y,z,vx,vy,vz]` and corresponding uncertainty.
- Project all selected points into each of the **eight cameras** at every frame; first depth wins per pixel, with the existing deterministic tie rule.
- Hard measured foreground masks: background raw RGB-D is **never** admitted as a valid temporal measurement. Uncovered foreground remains explicitly unexplained.
- Each point may be first hit in multiple cameras and may transition between them. Its **RGB-D temporal reference is camera-specific**; no canonical color comparison across different cameras.
- The previous image is sampled through the explicitly specified bilinear temporal warp. A new camera reference is acquired only when genuinely visible and accepted; there is no imaginary previous-camera reference.
- Current and reference point positions and their covariance remain coupled through the specified temporal likelihood and dynamics.
- A raw camera/frame/pixel RGB-D observation is a **single** uncertain latent measurement even if reused in multiple temporal interpolations. Correlations from that reuse are never arbitrarily discarded in `full` mode.
- Include the coupled source depth–backprojection prior only once; no second independent source XYZ prior.
- Support both `--temporal-warp-noise ideal` (EXACT zero *additional* warp discrepancy, while real sensor/reference/state uncertainties remain) and correctly specified nonzero warp discrepancy. Do not inject epsilon variance, diagonal jitter or fabricated sensor calibration.
- Keep `simplified` as an explicitly *approximate*, independent 6x6 point IEKF ablation. **Never substitute simplified for full**, even on resource failure.
- Preserve the original `legacy` estimator and existing downstream stages; do not silently change their mathematics.
- No ground truth in inference, sampling selection, sensor calibration, data association, noise, acceptance or process model. GT is for post-hoc validation only.

**Definition of “exact”:** the specified *locally linearized* fixed-lag Gaussian problem is solved and marginalized without dropping nonzero information terms, reference-pixel cross-covariance, or hard equality constraints. This does NOT claim a globally exact nonlinear Bayesian solution, nor claim the local image translation warp is a perfect physical model.

## 1. PIN AND INSPECT THE REAL CODE — BEFORE EDITING

1. Run `git status --short; git rev-parse HEAD; git branch --show-current; git diff --stat`. **Do not reset or overwrite uncommitted changes.** If the local tree differs from `f77492d`, inspect and use the actual local implementation; state the difference in the report.
2. Read, at minimum, these **actual committed source locations**:
   - `docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md` — declared likelihood, initialization, full covariance, FEJ, and limitations.
   - `estimator/temporal_rgbd_terms.py` — `temporal_prediction`, `build_temporal_rows`, bilinear weights and derivatives, foreground/reference checks.
   - `estimator/temporal_zbuffer.py` — `first_hit_pixel_owner`, incremental ownership and original-ID tie policy.
   - `estimator/temporal_reference.py` — per-point/per-camera reference lifecycle.
   - `estimator/temporal_sparse_gpu.py` — `UniquePixelTableGPU`, `TemporalSoAGraph`, `evaluate_factor_residuals_and_blocks_gpu`, `prior_innovation_coordinates_gpu`.
   - `estimator/temporal_sparse_solve.py` — `eliminate_current_pixel_constraints_gpu`, `assemble_augmented_kkt_gpu`, `SparseJointFactorization`, `extract_joint_selected_marginals_gpu`, `connected_columns_gpu`.
   - `estimator/temporal_cudss.py` — `CudaSparseFactor.__init__`, native memory/resource diagnostics, `solve`/refinement, `close`.
   - `estimator/temporal_sparse_nonlinear.py` — `snapshot`, `restore`, `refresh_reference_support_gpu`, `evaluate_nonlinear_objective_and_visibility_gpu`, `true_variable_block_components_gpu`, `solve_nonlinear_sparse_iekf_gpu`.
   - `estimator/temporal_sparse_marginalization.py` — exact sparse FEJ boundary Schur implementation.
   - `estimator/temporal_sparse_run.py` — `run_temporal_sparse_free`, complete post-acceptance covariance/publication path.
   - `scene/canonical_subsample.py` and `estimator/free_projective_run.py:build_catalogue` — existing deterministic seeded-uniform sampling.
   - `train.py`, `tools/run_temporal_rgbd_critical.py`, `tools/temporal_batch_runtime.py`, `utils/temporal_output.py` — CLI, deadlines, journaling, fail-fast, output validity.
   - `reports/temporal_sparse_gpu_exact_implementation.md` and `reports/temporal_sparse_gpu_validation.json` — latest failure, numerical witnesses and source hashes.
3. Produce a short `CURRENT_CODE_VS_EQUATIONS` table. Avoid implementing a second copy of a feature that already works.
4. Leave all previous failed artifacts intact. Do not launch another multi-scene/60-frame run.

## 2. THE MATHEMATICAL MEASUREMENT MODEL — NO DEVIATIONS

For point `i`, camera `c`, current frame `t`, last valid same-camera reference frame `r`:

```
s_i,t = [x_i,t, v_i,t] in R^6
s_i,t^- = F(dt) s_i,t-1^+
P_i,t^- = F P_i,t-1^+ F^T + Q(dt)

u_i,c,t = pi_c(x_i,t)
z_i,c,t = Z_c(x_i,t)
owner_c,t(p) = argmin_{j projecting to p} Z_c(x_j,t)
```

The currently first-hit point is the **only** point whose temporal observation may be assigned to pixel `p` in camera `c` (subject to valid foreground and reference support). Visibility is dynamic, recalculated after accepted corrections. Do not attach a point permanently to its originating camera.

Temporal warp and prediction (same `c` for reference and current):

```
q = p - pi_c(x_i,t) + pi_c(x_i,r)
DeltaZ = Z_c(x_i,t) - Z_c(x_i,r)
I_hat_i,c,t(p) = sum_{a=1}^4 w_a(q) ell_RGB,c,r,a
D_hat_i,c,t(p) = sum_{a=1}^4 w_a(q) ell_D,c,r,a + DeltaZ
r_i,c,t(p) = ell_clean,c,t,p - [I_hat, D_hat]
```

The image gradient comes from **the four raw reference pixels**, interpolated consistently, not from arbitrary other-camera RGB. Correct prediction Jacobians:

```
H_t^I = -grad I_ref(q) J_pi(x_t)
H_r^I = +grad I_ref(q) J_pi(x_r)
H_t^D = J_Z(x_t) - grad D_ref(q) J_pi(x_t)
H_r^D = -J_Z(x_r) + grad D_ref(q) J_pi(x_r)
H_(reference clean pixel a) = w_a(q) I_4
```

These are Jacobians of predicted `[RGB, D]`; residual Jacobians have the opposite signs for state/reference/prediction terms. All direct velocity columns of the RGB-D measurement are zero. Velocities are inferred by the coupled dynamical model. **No new direct optical-flow or rigid-part velocity measurement is to be added.**

Full likelihood, with each raw pixel included *once*:

```
y_c,k,p = ell_c,k,p + eta_c,k,p       eta ~ N(0, R_sensor,c,k,p)
ell_c,t,p - h(ell_c,r, x_i,t, x_i,r) = eps_warp
                                  eps_warp ~ N(0, R_warp)
s_i,k - F_k s_i,k-1 ~ N(0, Q_k)
```

`R_warp == 0` in diagnostic `ideal` mode is a **hard equality**, not a Gaussian with near-zero diagonal. Do not invent an error floor. For SPD `R_warp`, retain the specified stochastic factor. Exact raw-measurement ID: `(camera, frame, flat_pixel)`; index canonical point IDs separately from local array indices.

For the first anchor, retain the source-pixel coordinate error + clean source depth + velocity parameterization, including `Cov(x_source, ell_source.depth)`. Never include both this source model **and** an independent second `Pxx` likelihood. After linearization, exact marginal and cross-covariance come from the **joint** constrained posterior, not `inv(H_i^T R_i^-1 H_i + P_i^-^-1)` when there are shared observation correlations.

The full objective is a **fixed-lag** iterated linearized Gaussian objective, with FEJ (first-estimate Jacobian) linearized boundary information after exact Schur elimination; the fixed-lag approximation is declared. Hard visibility/observation support is piecewise constant during a local linearization. The nonlinear optimizer must explicitly rerender after proposals and must **never improve the objective by silently deleting difficult measurements**.

## 3. FIRST PRIORITY: DIAGNOSE THE *EXACT* SECOND FACTORIZATION FAILURE

The current measured evidence is: USB, **95,967 points**, all 8 cameras; initial frozen KKT factors and joint marginals are computable; the first nonlinear step lowers objective `44062961178.87926 -> 43743829644.68693`; then a second cuDSS factorization gives `INFO=-1`. The failed next system reportedly contains a true connected component of **371,164 scalar variables**. GPU is an **8 GB RTX 4070 Laptop**. Frame 1 was **not published**.

Do **NOT** assert `INFO=-1` proves singularity or OOM. Do **NOT** blindly increase the timeout or cuDSS capacity and rerun. Build a small, read-only, executable diagnostic that captures these stages without publishing an invalid frame:

- S0: initial accepted state/measurements and first assembled `A`, `C`, FEJ information, KKT; indexed key maps.
- S1: certified initial sparse solve, complete trial step and actual accepted objective/visible owner changes.
- S2: the **new relinearized** `A`, `C`, FEJ information and KKT which fails in cuDSS.
- For both S0 and S2 record all dimensions, `nnz`, independent real connected components, component scalar-size distribution, column/row degrees, factor types, raw-pixel ID counts, repeated measurement key counts, reference interpolation cells and weights, tie/collision counts, masks, exact constraints, removed/reused/added rows, and predicted symbolic fill.
- Record `torch.cuda.mem_get_info`, Torch allocated/reserved/max allocated, cuDSS native allocation estimates, `MAX_LU_NNZ`, `CUDSS_DATA_INFO` and exact call/status at failure, current stream and whether all prior native handles were properly destroyed. Separate CUDA allocation, native cuDSS allocation, insufficient configured factor capacity and numeric rank/conditioning explanations.
- `last_nonlinear_history` reports many frozen components; verify that `restore`, freshly registered raw reference pixels, acceptance/rebinding and `project_ideal_pixels_gpu` cause **no accidental duplicate measurement factors** or stale keys.
- Save a **reproducible, sparsely encoded diagnostic snapshot**, not a dense matrix. Preflight SSD space and use a compact non-destructive output directory. If the snapshot cannot be stored, save validated sufficient counts/indices and state precisely what was omitted. Provide a separate diagnostic command; no giant matrices printed to console.

### 3.1 “Connectivity” means algebraic dependence ONLY, never a motion graph

A factor can connect two physical points **only** through an actual nonzero derivative/coefficient of one of the permitted likelihood or Schur terms, most notably a reused uncertain raw pixel. The point state itself contains no neighborhood adjacency.

For each nonzero off-point coupling in S2, provide its provenance:

```
source factor(s): same raw (c, frame, pixel) used by multiple warp rows;
source backprojection/depth coupling; temporal process history; or prior FEJ Schur fill.
```

Verify all indices, cameras, frames, point IDs, interpolation weights and exact nonzeros; compare to an independently assembled CPU oracle on selected complete small components. **Never introduce point-to-point edges as a remedy.** `true_variable_block_components_gpu` may add *symbolic block connectivity for line-search consistency*, but those symbolic ties must **not** enter the Gaussian information matrix, and must never be misreported as measured statistical coupling.

Important distinction: as points acquire subpixel motion, originally zero interpolation weights may become nonzero, and reused reference pixels may legitimately couple point states. Confirm this numerically rather than just asserting it. Any additional coupling due to erroneous keys/stale binding is a bug to remove surgically.

Tests for this priority:

1. Every numerical off-point nonzero is traceable to an analytical factor or an induced FEJ Schur term.
2. No cross-camera raw-pixel identity collision; key encoding and `original_source_ids` are injective and stable.
3. No duplicate sensor likelihood on a reused raw pixel.
4. Line-search *symbolic* block ties do not appear in the actual information matrix.
5. True connected-component extraction matches an independent CPU union/find over **actual nonzero coefficients** on deterministic toy + sampled real small complete components.
6. The S0-to-S2 change in giant component size is accounted for by actual, numerically nonzero factors (no threshold-to-zero pruning; exact zero vs nonzero is reported).

## 4. SECOND PRIORITY: COMPLETE THE REAL SPARSE GPU SOLVE WITHOUT CHANGING THE MODEL

The current exact GPU math is substantially implemented. **Do not rewrite the entire pipeline.** Improve `estimator/temporal_cudss.py`, `estimator/temporal_sparse_solve.py`, and only the necessary supporting sparse layout code.

### 4.1 Required numerical semantics

The actual objective stays (up to constants):

```
min_delta 0.5 ||A delta + r||^2 + 0.5 delta^T H_FEJ delta + g_FEJ^T delta
subject to C delta + c = 0.
```

The current innovation-coordinate transformation, analytic substitution of `+I4` clean-current-pixel ideal constraints and exact source-depth correlations are valid **if** they pass parity. Preserve them. Any alternative sparse ordering, exact block decomposition or direct factorization must produce the same constrained MAP solution and same queried **joint** covariance of the local linearized model.

### 4.2 Storage/factorization requirements

- Analyze the S2 matrix *before* factorization. If the operator is block diagonal after exact permutation, factor its **genuinely disconnected** blocks independently. This is an exact algebraic reordering, **not** an approximation or point grouping. Include all cross-block edges from source pixels, temporal factors and FEJ before deciding independence.
- Independently check whether the reported 371,164-variable block is intrinsic or inflated by numeric key/assembly/ordering issues. Correct only proven bugs.
- Factorization workspace/capacity must be based on actual S2 symbolic analysis, not an arbitrary `MAX_LU_NNZ` guess. Distinguish host heap, device heap, cuDSS internal/native memory, Torch live tensors and fragmentation. Document the chosen cuDSS algorithm/ordering and their memory support. Avoid device memory leaks between Newton iterations.
- If a **mathematically identical** cuDSS configuration or another mature sparse CUDA backend provides a more memory-efficient exact direct factorization, implement it after independent oracles pass. Do not claim selected backend supports a mode it does not. Keep FP64. Do not form an approximate diagonal Schur, drop weak modes, threshold small covariance terms, invent a ridge, break connected components, or change process/reference noise.
- If exact full-cloud S2 cannot fit on the 8 GB GPU despite correct storage/lifetime/reordering, emit `FULL_RESOURCE_LIMIT` with the measured requirement and provide an **otherwise identical run command** for a 32 GB GPU. Do not call a resource limitation a failed physical theory. Do not silently fall back to CPU, simplified inference or subsampled points.
- Avoid a new homegrown sparse direct factorization if a verified production library exists. Use library error codes literally; `INFO=-1` requires precise interpretation from the installed ABI/docs and independent numeric checks.

### 4.3 Joint posterior covariance is a mandatory part of the solve

The frame cannot be published after MAP alone. The final *accepted* relinearized system must support:

```
s_t^+       = accepted/current global states
P_i,t^+     = Q_i Cov_joint Q_i^T       for all selected points i
Cov(s_i,t, s_i,r) and Cov(s_i,t, ell_c,r,p) on requested diagnostic queries
```

Use selected solves on the same fully correlated constrained sparse factorization. Preserve cross-time and cross-point dependencies. Verify symmetry, PSD/positive-definiteness where theoretically applicable, exact hard-constraint null directions when querying constrained variables, and numerical agreement against independent dense CPU QR/SVD oracles on **complete** small components.

After each accepted update: final relinearization -> constrained solve/certificate -> all-point selected covariance -> output validity checks -> publish frame journal -> reference refresh. The runner's existing transaction and NaN-tail implementation must remain; never publish the intermediate accepted step as a complete posterior.

If the mathematical factorization itself cannot be certified on given hardware, stop with an explicit failure report containing resource and solver certificates. Never produce a zero-filled or fake trajectory.

## 5. THIRD PRIORITY: ENSURE THE NONLINEAR STEP MATCHES THE ACTUAL MEASUREMENT MODEL

Focus `estimator/temporal_sparse_nonlinear.py` and `estimator/temporal_sparse_run.py`; preserve the existing full observation model.

- The IEKF prediction prior is fixed during inner iterations. A newly computed Jacobian and depth ownership can be used on accepted relinearization, with physically valid common-domain objective handling.
- Rerender first-hit owners after proposed changes. A current pixel cannot be claimed by multiple points, nor can it be silently removed from the objective just because the prediction moved away.
- Reference interpolation can move to a different cell. Add the true newly used raw pixels **once** and correctly account for their independent raw sensor likelihood and the induced cross-covariances. Do not freeze reference cells globally or treat their weights as constant across accepted iterations.
- Strict reference validity and foreground masks must remain. Do not introduce new velocity clamps, association gates, deformation priors, static/moving labels, arbitrary displacement bounds, or hard-coded “reasonable” motion thresholds.
- Explicitly compare analytic relinearized objective to a recomputation from the same real raw pixels and the current depth-sorted ownership; include actual hard-equality residuals. Never accept on an obsolete domain or stale reference key.
- Preserve the exact distinction between a **no-direct-measurement point** and a point which can still receive information through shared observation cross-covariance.
- An optimizer that freezes every affected component or only reports accepted state changes, but never finishes a certified posterior, has **not passed** the tracking acceptance gate.

Test static scene, moving textured patch, known translation, true subpixel motion crossing interpolation cell, two points sharing an uncertain reference pixel, occlusion switch, camera-A-to-B handover, and a locally rotating articulated part. For the last case **evaluate and report limitations** of local translation warp; do not invent a rotation/patch-orientation state or change the prescribed warp.

## 6. FOURTH PRIORITY: EXACT LAG SLIDE WITH THE SAME GPU SYSTEM

In `estimator/temporal_sparse_marginalization.py`:

- Eliminate exactly the expired variables from the specified **linearized** joint Gaussian model, retaining complete induced information/gradient, FEJ linearization, residual constants and surviving hard equalities.
- Do not drop nonzero fill-in; do not split a coupled factor by point; no diagonal approximation to marginal precision; no covariance replacement with pointwise 6x6 independent blocks.
- Verify retained mean, selected covariance, current/reference cross-covariance, and hard equalities **before versus after** marginalization on small complete models and actual small real components.
- Test a valid camera reference that survives the lag and a stale one that must bootstrap again; forward/backward frames with actual signed timestamps.
- The real full-cloud minimum final gate needs **4 original frames with lag 3**, because 3 frames alone do not establish a real lag-boundary elimination. Keep this as a separate gate only after frames 1 and 2 work.

## 7. UNIFORM POINT DOWNSAMPLING — EXISTING, OPTIONAL, NEVER HIDDEN

**Already implemented:** `scene/canonical_subsample.py:subsample(...,method='seeded-uniform',seed=...)` selects `ceil(N_full/factor)` original canonical IDs without replacement, sorted to preserve point-ID order. `train.py` already supports `--pcd-subsample-method seeded-uniform --pcd-subsample-factor N --pcd-subsample-seed S`. Reuse this code; **do not invent another sampler**.

**Remaining bug/feature:** `tools/run_temporal_rgbd_critical.py` currently forwards only `--pcd-subsample-factor`, not `--pcd-subsample-method` or `--pcd-subsample-seed`.

Implement **only**:

1. Add those two batch options with the same semantics as `train.py`; pass all three flags through exactly to every selected child, including full/simplified/legacy comparison modes.
2. Confirm selection occurs once before source initialization, follows canonical original IDs throughout the frame sequence, and does **not** reduce the eight camera observations, image resolution, number of frames or any numerical covariance term in the selected point set.
3. Confirm the first-hit renderer, source pixel IDs, unique pixel registry, temporal reference states, sparse factor shapes, Schur, covariance queries, outputs, debug traces and GT **post-hoc** evaluator all use the selected point-ID map consistently.
4. Test `factor=1` bitwise identity; deterministic `factor>1` (e.g. 2/4/8) for equal seeds; different seeds select corresponding valid original subsets; no replacement or ID renumbering; exact full mode for the *selected* cloud. All downsampling reports must include `N_full,N_selected,factor,method,seed,selected_ids_sha256` and the unchanged original point IDs.
5. Explicitly compare owner maps against factor=1: removing occluders changes visible owners, so downsampled tracking cannot be presented as an unchanged full-cloud inference result.

Keep factor=1 the production default. Use factors 8 then 4 as convenient *diagnostic* scaling gates on the SAME full sparse implementation; they **never** replace the required factor-1 USB/Blade full-cloud gate. The label `full` identifies exact covariance **conditional on the selected cloud**, not a claim all points were included.

## 8. RUNTIME AND OUTPUT SAFETY — KEEP AND VERIFY, DO NOT REIMPLEMENT

The latest commit added these and 60 tests passed: hard OS process-group deadline, fail-fast batch, transaction journal, NaN tails for unwritten frames, disk-space preflight and incomplete-file refusal. **Preserve them**.

- All stage failures include scene, mode, original frame index, completed and valid frames, exact cause, timing, GPU/Torch/native allocation info, measured matrix shape/nnz and solver status.
- A valid frame requires accepted state, final covariance, complete data arrays and debug/status publication; commit journal is authoritative. Frame-0 geometry alone is not a successful tracking run.
- `--continue-on-error` remains explicit; default is fail-fast. No whole 10-scene attempt until the gates below pass.
- Do not automatically rerun a known failing system for hours. Respect the 120s bounded child deadline and leave a concise failure artifact on resource errors.
- Do not write a source snapshot inside a large SSD run, or delete historical results without instruction.

## 9. MANDATORY TEST GATES — FAIL THE TASK IF ANY REQUIRED GATE FAILS

### G0 — Mathematical identity and original legacy parity

- Run existing `python -B tools/test_temporal_rgbd.py`. Preserve the 60 passing tests, plus new targeted tests; any failure must be explained/fixed surgically.
- Verify numerical Jacobians against central differences on smooth interpolation cell interiors, using both reference and current 6D states. Test exact zero velocity Jacobian columns.
- Confirm `full` and `simplified` use the same visibility/temporal row operator. Confirm `legacy` arithmetic is unchanged.
- Verify no new spatial/motion graph connectivity or priors exist by introspecting actual assembled columns and sparse factors.

### G1 — Independent exact small-system certification

- Construct CPU float64 **independently coded** dense whitening/constraint reference for tiny complete cases; compare objective, MAP correction, hard equality residual, selected joint covariance, source-depth cross covariance, pixel-reuse cross-point covariance and FEJ Schur before/after.
- Compare `CudaSparseFactor` on the identical actual sparse matrices and truly connected components. Certified residuals must use the original problem and fixed mathematical error tolerances, not just cuDSS success codes.
- Include a case where two points reuse one raw RGB-D reference pixel and a case where they do not; information coupling must appear/disappear accordingly.
- Include a connected-component extraction reference oracle built from actual sparse nonzero entries, with no synthetic motion edges.

### G2 — Reproduce and resolve the USB SECOND FACTORIZATION failure

- Capture S0/S1/S2 traces from the **unchanged real input** with all 95,967 points / eight cameras / no subsampling and reproduce the initial accepted objective drop.
- Prove why the next operator differs and where 371,164-variable coupling originates. Diagnose `INFO=-1` from real library data and memory reports, not guesses.
- With a corrected exact solver, compute the second solution and selected covariance. If it cannot fit in 8 GB, record a hard, measured `FULL_RESOURCE_LIMIT` and reproduce on a larger, available GPU using the identical factor=1 equation. This is a hardware limitation, not a license to corrupt the model.

### G3 — TRUE estimated full-cloud frames 0,1,2 (not a frozen linearization)

- Run USB factor=1, all eight cameras, forward, 3 frames; then Blade factor=1 only after USB fully passes; use a hard child-process timeout and fail-fast.
- Frames 1 and 2 must be **journal-verified, finite, nonzero only where genuinely estimated, with complete 6x6 joint marginals** and meaningful nonzero measurement records/accepted corrections. Test masked regions, physical prediction, per-camera references, and condition of covariance.
- Verify that frame-1 posterior was calculated from the FINAL accepted relinearization, not from the pre-step sparse factorization.
- If `G3` fails, halt longer validation and report failure. **Do not claim completion from one accepted intermediate correction.**

### G4 — REAL lag slide, backward pass, camera handover

- On one manageable genuine scene, run factor=1 4 frames with `lag=3`; ensure one actual FEJ marginalization, preserved cross-covariance and a valid new posterior. Repeat short backward pass from the correct endpoint; compare local numerical oracles.
- Test reappearance/first acquisition with camera-specific temporal references; first B observation may bootstrap without a temporal RGB correction, later B observations may update. Do not require source camera to remain visible.

### G5 — Scientific validity, no GT leakage

- After G3/G4: evaluate short USB, Blade, Storage and Stapler motion using external synthetic GT only in independent evaluation. Compare same-frame, same-IDs, same cameras and point populations with legacy and simplified; report moving/static point position error, support/owners, loss and gains. Preserve the original failure-point traces (Storage 8444,45588; USB 67827,87290,85696; Blade 27016).
- Test the temporal warp **at GT point trajectories offline**, as a falsification test for the local-translation RGB-D observation. If the warp model is invalid even at true motion, stop and report `TEMPORAL_WARP_MODEL_INVALID` with evidence; **do not silently invent a different method**.
- No uncalibrated ideal-warp result may be described as physically calibrated uncertainty.

### G6 — Scaling and repeatability (only after gates above)

- On one bounded synthetic scene, run factors 8/4/2/1, same `seed`, all eight cameras, same frames, **same full solver**, record runtime breakdown (GPU assembly, symbolic analysis, numeric factor, nonlinear trials, covariance, Schur, exports), host and native GPU memory, connected-component distribution and filtering accuracy.
- Stage results by `factor` and `original-ID` mapping. Do not compare populations without matching IDs; never claim factor=8 establishes factor=1 feasibility.
- Do not run all 10 scenes/60 frames until G3, G4, and a smaller scientifically meaningful accuracy pilot have passed.

## 10. EXACT RUN COMMANDS (FOR USER — DO NOT AUTO-LAUNCH A LARGE BATCH)

Check local tree and focused tests:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
git rev-parse HEAD
git status --short
python -B tools/test_temporal_rgbd.py
```

After Codex implements full batch forwarding for method/seed, these are the required limited gates:

```bash
# Uniform point subset; mathematical full inference on selected points only.
bash tools/run_temporal_rgbd_critical.sh \
  --mode full --direction forward --frame-limit 3 \
  --scenes USB_100109 \
  --pcd-subsample-factor 8 \
  --pcd-subsample-method seeded-uniform \
  --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal \
  --stop-after free --temporal-runtime-budget-seconds 120
```

```bash
# Mandatory original full cloud (after the S2 memory defect is resolved).
bash tools/run_temporal_rgbd_critical.sh \
  --mode full --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 \
  --pcd-subsample-factor 1 \
  --pcd-subsample-method seeded-uniform \
  --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal \
  --stop-after free --temporal-runtime-budget-seconds 120
```

**These commands do not mean a successful full-cloud GPU solve is currently possible on the 8 GB device.** If there is a demonstrated exact solver resource limit, Codex must supply the *unchanged* command/environment for an available higher-memory GPU rather than weaken the mathematical model.

## 11. REQUIRED FILES / ACCEPTANCE EVIDENCE

Deliver actual source edits (not stubs) and a compact report at `reports/temporal_sparse_gpu_completion.md` including:

1. Verified local HEAD and changed source files/functions. Each change tied to a proven defect or needed missing feature; preserve baseline and previous runs.
2. `MODEL_PARITY`: equations -> concrete implementation -> independent numerical witness with exact 6x6 and full joint covariance output.
3. `S2_FAILURE_CAUSE`: initial and relinearized sparse system differences; exact provenance of any cross-point coupling; status/estimated native memory/capacity/VRAM at failure; observed fix or explicit resource limitation.
4. `GPU_SOLVER_CERTIFICATE`: original-system numerical residual, equality residual, objective, selected joint covariance and FEJ Schur error vs independent dense oracles, plus selected real complete components.
5. `ACTUAL_FRAME_EVIDENCE`: verified frame journal and per-frame state/covariance filenames; frame-1 and frame-2 nontrivial estimates, true accepted measurements, occupancy/reference history, raw residuals and full/selected sensor noise provenance. Frame-0-only is FAILURE.
6. `DOWNSAMPLING_VALIDATION`: factor=1 parity, deterministic ID selection, method and seed forwarding, benchmark and changed-visibility report.
7. `RESOURCE_PROFILE`: peak native/Torch/host allocations, full matrix/true-component nnz, factorization/Schur and covariance time, infeasibility/timeout rationale.
8. `REGRESSION_REPORT`: focused tests + original baseline suite outcomes separated, with failures traced to exact file/function.
9. `EXACT_USER_COMMANDS`: bounded and verified; no fabricated paths, unimplemented flags, or false speed claims.
10. Explicit status `FULL_SPARSE_GPU_ACCEPTED` ONLY if all G0–G4 pass, including full cloud USB and Blade frames 0–2 with complete joint covariance. Otherwise `FULL_SPARSE_GPU_NOT_ACCEPTED` with precise reason. Scientific validity is a separate status until G5 passes.

## 12. ABSOLUTE PROHIBITIONS

No placeholders, empty `pass` functions, fake “success” returns, dummy Jacobians, zero-filled future frames, silently skipped covariance publication, guessed parameter values, disabled tests, hidden exception swallowing, approximate independent updates in `full`, arbitrary numerical thresholds to remove connections, arbitrary covariance inflation, GPU-only labels for Python-per-factor implementations, added normal/mesh/surfel/Gaussian/rigid-motion states, externally imposed motion priors, or fabricated experimental results.

Do not spend hours repeating known failure cases without a new falsifiable change. Do not claim `INFO=-1` is singularity/VRAM without isolating the actual cause. Do not redesign the physical observation model if its correctness has not yet been tested against GT; report a model-validity failure and await a separate theoretical decision.

**Finish the implementation, not just a report. A bounded, actual estimated frame with a valid joint covariance is the minimum evidence that the implementation has progressed.**
