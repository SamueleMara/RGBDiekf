# Four-pass projective RGB-D IEKF: implementation report (2026-10-28)

**Contract:** `CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md`.

**Base:** HEAD `34131bcdcedbbd02f6118f1a766fd8394bed3b3e`, plus the uncommitted 2026-10-27 restructure.

**Snapshot before editing:** `/home/samuelemara/Joint3DGS/RGBDekf_snapshot_before_projective_20261028.tar.gz`. The migration audit is in [projective_rgbd_migration_audit.md](projective_rgbd_migration_audit.md).

**Nothing is committed.**

**Status: NOT complete in the contract's sense.** All four filter passes and every downstream stage are implemented, run end to end on real data, and pass their numerical gates:
- Kalman algebra;
- Jacobians;
- Lie conventions;
- sensor sampling;
- visual QA.

The method does not yet produce correct articulation results:
- On no real scene does the joint decision accept the joint.
- The free filter cannot follow motions larger than about 1–2 px per frame.
- The rigid covariance is overconfident.
- A static scene yields a false joint.

The details and evidence follow.

## 1. Files

**Added, production:**

| Group | Files |
|---|---|
| Entry points | `train.py` (four-pass driver), `tools/run_rgbd_projective_pipeline.py` (contract name, same code) |
| Configuration | `configs/projective_rgbd.json` (declared policy) |
| Estimator | `estimator/projective_camera.py`, `projective_sensor.py`, `projective_visibility.py`, `projective_terms.py`, `point_kalman_cuda.py`, `free_projective_run.py`, `projective_coherence.py`, `se3.py`, `rigid_kalman_cuda.py` |
| Scene | `scene/canonical_subsample.py` |
| Joint and support | `joint_reduction/joint_projective.py`, `support_projective.py` |
| Debug pages | `debug/projective_debug.py`, `debug/assets/projective_common.js`, `projective_{free_3d,free_rgbd,coherence,rigid_3d,rigid_rgbd,joint,bidirectional,numeric,index}.html` |
| Tools | `tools/evaluate_projective_gt.py`, `tools/qa_projective_debug.py`, `tools/profile_rgbd_projective_cuda.py` |

**Added, tests:**
- `tests/projective_fixtures.py`
- `test_projective_camera.py`, `test_projective_measurement.py`
- `test_point_kalman_cuda.py`, `test_rigid_kalman_cuda.py`
- `test_canonical_subsample.py`, `test_joint_projective.py`
- `test_four_passes.py`

**Moved:** the previous `train.py` is now `tools/legacy_rendered_splat_train.py`, labelled `LEGACY_RENDERED_SPLAT_BASELINE`.

**Reused unchanged:**
- scene loading (`scene/*`), endpoint back-projection (`scene/endpoint.py`);
- the pinned coherence extractor (`estimator/motion_coherence_support.py`, `estimator/supported_coherence.py`, `rigeuler_ref/coherence_extraction.py`);
- the pinned `rigeuler_ref/rigid_trajectory_reduction.reduce_trajectory`;
- `joint_reduction/bidirectional.line_distance`.

**No production dependency on the retired model.** This was checked on the import closure of `train.py`, `debug/projective_debug.py`, `tools/evaluate_projective_gt.py` and `tools/qa_projective_debug.py`. None of the following is reachable from it:
- `point_renderer` (splats, `render`), `observation_terms` (Student-t, coverage energy, `FrameObservation.terms`);
- `sparse_iekf` (trust region, PCG), `covariance`, `restricted_posterior`;
- `supernodes.GroupBasis` (the `G ζ` tangent), `phase1`, `rigid_part_iekf`, `joint_iekf`;
- `joint_reduction/support.py` (Gibbs energy, ICM), the debug renderers, and any association module.

The only kNN is the pinned coherence graph over the 192 representative nodes, which the contract mandates (§4.3). It is not a correspondence. One leftover remains: `scene/config.py` still loads the old association tail probability. Its value is never used by the projective pipeline.

## 2. Equations as implemented (conventions included)

**Geometry** (`projective_camera.py`):
- `T_cw` is world→camera, `X_c = R x + t`, and `T_wc = T_cw⁻¹` is persisted.
- `u = f_x X/Z + c_x`, `v = f_y Y/Z + c_y`.
- `J_π = [[f_x/Z, 0, −f_x X/Z²], [0, f_y/Z, −f_y Y/Z²]]`, `J_u = J_π R`, `J_Z = e₃ᵀR`.
- `u` is the column and `v` the row; the centre of pixel (r, c) is at (c, r).
- The rank-one identity `x_meas − x_pred = Rᵀ b (D − Z)` is tested to 1e-12.

**Measurement** (`projective_sensor.py`, `projective_terms.py`):
- Bilinear interpolation of the measured fields with its exact gradient: `∂F/∂u = (1−dv)(F₁₀−F₀₀) + dv(F₁₁−F₀₁)`, and likewise for v.
- `h = [I(π) − a; D(π) − Z]` and `r = −h = [a − I; Z − D]`.
- `H_x = [∇I J_u; ∇D J_u − J_Z]`; the velocity columns of H are zero.
- `R = diag(Σw² σ_rgb², Σw_j² σ_D(D_j)²)`. The σ values come from `configs/example_uncalibrated.json`, which declares itself uncalibrated (σ_rgb = 0.05, σ_D = 1 cm); model-mismatch variance is 0, declared.
- A sample is valid only if all 4 neighbours are inside the image, the mask and valid depth, and lie on one surface: `spread ≤ 3√2 σ_D + z·tan 80°·√2/f`.
- **Pixel reuse:** within each pixel, only the front-most point (smallest Z, then smallest id) uses the sample. The others are `SELF_OCCLUDED` if they lie more than `3√(σ_zi² + σ_zf²)` behind it, else `DUPLICATE_PIXEL_SUPPRESSED`.
- **External occlusion:** a row with `Z − D > 3√S_zz`, where `S_zz = H_D Σ_x H_Dᵀ + R_D`, is `MEASURED_SURFACE_IN_FRONT` and is reported but not used. The opposite sign is kept.

**Free 6D IEKF** (`point_kalman_cuda.py`):
- **Prediction:** `F = [[I, Δt I], [0, I]]` and `Q = q_c ⊗ [[τ³/3, sgn τ²/2], [sgn τ²/2, τ]]` with `τ = |Δt|`. `q_c = 0.01 m²/s³` is the declared, uncalibrated white-acceleration density; the contract has no position random walk, so none is used.
- **Initial state:** measured endpoint `x` with its back-projection covariance; `v = 0` with `P_vv = 1 m²/s²`.
- **Update:** fixed `(s⁻, P⁻)`; at each iterate:
  - `A = Λ⁻ + Σ_c H_cᵀR_c⁻¹H_c` and `b = Σ_c H_cᵀR_c⁻¹r_c − Λ⁻(s_j − s⁻)`;
  - `δ = A⁻¹b` by a batched Cholesky solve;
  - `s_{j+1} = s_j + δ`.
  This equals the textbook iterated update `s⁻ + K[r + H(s_j − s⁻)]` (dense test, 1e-10).
- **Stopping:** each point stops by itself when `δᵀAδ < 1e-8`, at most 10 iterations.
- **Posterior:** once, at the final iterate's own linearization, `P⁺ = (Λ⁻ + HᵀR⁻¹H)⁻¹`. It matches the explicit Joseph form `(I−KH)P⁻(I−KH)ᵀ + KRKᵀ` to ≤5e-13 in the blade traces.
- **No valid row at the final linearization:** `s⁺ = s⁻`, `P⁺ = P⁻`.
- **NIS:** at the prior, `rᵀR⁻¹r − gᵀA⁻¹g`.
- Points are independent: there is no group and no shared increment (test: a point's result is identical when filtered alone, to 1e-12).

**Rigid 12D IEKF** (`se3.py`, `rigid_kalman_cuda.py`):
- **Convention:** `ξ = [v, ω]`, spatial left `T' = Exp(δ^)T`, `x = R p + t`, `T_anchor = I`. `Exp` is the matrix exponential, `J_l = Σ adⁿ/(n+1)!`, and `Ad = [[R, t^R], [0, R]]`.
- **Prediction:**
  - `T⁻ = Exp(Δt ξ) T`;
  - `F = [[Ad_E, Δt J_l(Δt ξ)], [0, I]]` (finite-difference tested for ±Δt);
  - `Q` = integrated white twist acceleration, density `Q_c = process_covariance_rate[6:12, 6:12]`, first order in the pose block (stated).
- **Measurement:** the same `build_terms` rows, chained by `B_i = [I, −[x_i]×]` to give `H_rigid = [H_x B, 0]`. Static-compatible points act only as occluders.
- **Update:**
  - `e = [Log(T T⁻⁻¹); ξ − ξ⁻]` and `L = blkdiag(J_l(e_τ)⁻¹, I)` (finite-difference tested);
  - `A = LᵀΛ⁻L + ΣHᵀR⁻¹H` and `b = ΣHᵀR⁻¹r − LᵀΛ⁻e`;
  - `T ← Exp(δ_τ)T`, `ξ ← ξ + δ_ξ`.
- **Covariance-chart policy: final-chart.** `P = A⁻¹` at the accepted iterate and no reset is applied. Checked against the local-chart Joseph form with `P_loc = L⁻¹P⁻L⁻ᵀ` to ≤6e-16 on the blade. The reset Jacobian `J_l(δτ)` was also verified numerically, although it is not used.

**Joint** (`joint_projective.py`):
- The pinned reduction runs in processing order, because it takes index 0 as the gauge.
- **Decision** at the declared α = 0.01, with the pose marginals treated as independent across frames (stated approximation):
  - static: `Σ Log(T_k)ᵀP_k⁻¹Log(T_k) ~ χ²(6(K−1))`, excluding the anchor;
  - prismatic and revolute: left-residual energies against χ² with `6(K−1) − n_par − (K−1)` degrees of freedom;
  - nested likelihood-ratio test `E_P − E_R ~ χ²(2)`, since prismatic is the infinite-radius limit of revolute.
- **Optional `(q, q̇)` IEKF:** `x(q) = T_J(q)p` with `∂x/∂q = e` (prismatic) or `e × (x − a)` (revolute), using the same measurement.

**Support** (`support_projective.py`): per frame, the mean over cameras usable under both hypotheses of `(cost_static − cost_moving)`. A measured surface in front of a hypothesis gives no evidence. Labels come from frame-level HAC one-sided tests with BH-FDR.

### Decisions where the contract left room (all stated in the code)

1. **Pixel-reuse assignment** (`--pixel-reuse-assignment`, default `per-frame`). Item 2.2(5) prescribes front-most-per-pixel reuse control but not when to evaluate it; the §7.2 pseudocode recomputes visibility every iteration. The default decides the assignment once, at the prediction, and still re-evaluates sampling, validity and Jacobians at every iterate. On the ray-cast fixture (9,300 textured points, 2.5 px prior offset):

   | Assignment | Median error | Points not converged |
   |---|---|---|
   | per-iteration | 6.84 mm | 3,815 |
   | per-frame | 0.68 mm | 1,136 |
   | duplicate suppression off (not allowed; reference only) | 0.51 mm | 1,414 |

   `per-iteration` remains available.
2. **No line search or damping in either IEKF.** This follows the contract ("not a hidden optimizer"). Non-converged points keep their last iterate, as a standard IEKF does.
3. **Coherence driving policy `pinned_raw`.** Both policies are always run and reported.
4. **Joint decision:** the anchor is excluded and the nested likelihood-ratio test is used, as above.

## 3. Gates

| Gate | Evidence | Result |
|---|---|---|
| **A** geometry | Projection and `J_u` against a NumPy oracle (CPU and CUDA, rotated and translated cameras, unequal f_x/f_y): ≤1e-9 px. Back-projection round trip and rank-one identity: 1e-12. Camera-to-world pose inverted once. Integer millimetre depth refused. | PASS |
| **A** measurement | Sampling against the CPU oracle: 1e-12. Full 4×6 H against finite differences: <1e-5 relative inside cells (blade traces 1e-10). `r = −h` exactly. Mask, invalid depth, depth break, out of image, behind camera, duplicate, self-occluded, external occluder and missing camera all produce no contaminated rows. Textureless plane gives rank-one depth information. | PASS |
| **B** Kalman | Dense iterated K = information form (1e-10). Joseph = information covariance (1e-9 dense; ≤5e-13 blade). Signed-time F and Q PSD. Prediction-only point keeps the prior bitwise. Depth-only plane: lateral variance unchanged (1%), normal contracted. Points independent (1e-12). CPU = CUDA (1e-10). | PASS |
| **C** free passes | Independent `P_0` / `P_T`, processing orders and chronological storage. Ray-cast plate at 0.8 px/frame: moving error 0.8 mm over 37.6 mm. Temporarily occluded static points held (<2 mm). | PASS on fixtures; real data see §5 |
| **D** subsampling | Factor 1 is the identity. Factors 2/4/8 select exactly `ceil(N/f)` existing ids, deterministically, with 0 voxel components lost; 378k points in 1.1 s. | PASS |
| **D** GPU | Profiles in §4. Storage (346k–378k points) completes within 1.9 GB. | PASS (measured) |
| **E** coherence | Pinned extractor unchanged (`test_coherence_parity` 4/4). Seed and recruited nodes saved separately; both weightings reported. | PASS (implementation) |
| **F** rigid | SE(3) maps against scipy and finite differences; F, L, B and J_reset by finite differences. Ray-cast prismatic, revolute and static scenes, both directions: pose error <3 mm / 0.3°. Pairwise rigidity <1e-12. Final chart = local Joseph. | PASS on fixtures |
| **G** joint logic | Calibrated synthetic pose sequences: STATIC, PRISMATIC and REVOLUTE correct in both directions; weak excitation not called revolute. | PASS |
| **G** static image scene | `test_static_scene_identifies_no_joint`: forward REVOLUTE, backward PRISMATIC (§5). | **FAIL** |
| **H** visual + numeric | Headless Chrome, all pages, both directions, frames 0/1/5/10/last. Markers = π(T_cw x) by NumPy (≤1.5e-5 px). RGB underlay = source PNG (0 of 50 pixels differ, per page). Selectors and sliders change the data. Backward pages label frame T as canonical. Screenshots inspected. | PASS (82/82 checks per run) |

**Test suite** (`python -m unittest discover -s tests -t .`): 99 tests, 93 pass, 6 fail.
- New projective tests: 40, with 1 failure (the static scene).
- The other 5 failures are the earlier rendered-splat tests (`test_image_rigid_joint` ×4, `test_image_support` ×1). They predated this work with identical values (`reports/restructure_test_comparison.md`).

## 4. Performance

Steady-state medians, RTX 4070 Laptop 8 GB. Full tables: [projective_profile/](projective_profile/).

| Scene | Factor | Points | Free frame (ms) | Peak GPU (MB) | Full run (both directions, all stages) |
|---|---|---|---|---|---|
| blade | 1 | 34,369 / 38,622 | 164–203 | 208–234 | 179 s (debug full, evaluation and QA included) |
| blade | 2 / 4 / 8 | 17k / 8.6k / 4.3k | 143 / 135 / 130 | 146 / 107 / 85 | 74 / 71 / 70 s |
| storage | 1 | 346,437 / 378,098 | 1,000–1,066 | 1,692–1,830 | 284 s (debug light) |
| storage | 2 / 4 | 173k / 87k | 555–605 / 330–353 | 958–997 / 595–611 | 155 / 114 s |
| laptop | 1 / 4 | 146k / 36k | 264–416 / 155–180 | 475–746 / 193–289 | 155 / 78 s |

Notes:
- **Stage split, storage factor 1:** visibility 241 ms, solve 193 ms, sampling 111 ms, accumulation 97 ms, image transfer 84 ms. On the blade, about half of each frame is decoding and transferring the 8 PNG and NPY files from the external SSD.
- **Two bitwise-identical optimizations** were made after the first profile and verified on every saved array:
  - visibility no longer computes projection Jacobians;
  - stopped points are not re-sampled or re-solved.
  Storage went from 1.67 to 1.0 s per frame, blade from 0.25 to about 0.15–0.2 s.
- **Precision:** float32 and mixed agree with float64 at the median (≤4e-7 m at factors 1–2). Single non-converged points end far apart (max 1–2 m) because their oscillation is chaotic. Float64 is kept as the default; no precision claim is made.
- cuSOLVER's batched `eigvalsh` needs about 0.5 MB of workspace per matrix and fails above about 8k matrices. Eigen diagnostics therefore use a device Jacobi solver; positive definiteness is checked by Cholesky on every point, and 6×6 eigen quantiles come from a deterministic sample of ≤16,384 points.
- Triton / custom CUDA kernels are **not implemented** (`--cuda-backend triton` is refused).

## 5. Results

Ground truth is post hoc, in `evaluation_gt/`. Runs are in `/home/samuelemara/Joint3DGS/RGBDekf_local_runs/projective_20261028/`.

| Run | Dir | Free moving error / true motion (m) | Free static error (m) | Coherence pinned_raw precision / recall | Rigid final error (m, °) | Joint |
|---|---|---|---|---|---|---|
| blade f1 | F | 0.60 / 0.835 | 0.119 | 0.76 / 0.91 | 0.035, 3.1 | UNIDENTIFIABLE (axis 0.6°, q RMSE 63 mm) |
| blade f1 | B | 0.66 / 0.835 | 0.118 | 0.83 / 0.31 | 0.0055, 0.6 | UNIDENTIFIABLE (axis 0.16°, q RMSE 3.1 mm) |
| storage f1 | F | 0.20 / 0.532 | 0.031 | 0.98 / 0.64 | 0.48, 92 (diverged after frame 43) | UNIDENTIFIABLE |
| storage f1 | B | 0.35 / 0.532 | 0.030 | 1.00 / 0.81 | **0.0016, 0.04** | UNIDENTIFIABLE (axis **0.11°**, q RMSE **0.4 mm**) |
| laptop f1 (revolute) | F | 0.93 / 0.57 | 0.122 | 1.00 / 0.68 | 0.40, 16 | UNIDENTIFIABLE (axis 0.07°, line distance 6.5 mm) |
| laptop f1 | B | 1.11 / 0.50 | 0.116 | 0.72 / 0.32 | 4.7, 110 | UNIDENTIFIABLE |
| static ray-cast | F / B | — | — | resolved a core on a static scene | — | REVOLUTE / PRISMATIC (**false**) |

**Subsampling trade-off.** Factors 2/4/8 cut the free-pass time only modestly on the blade, because image I/O dominates; on storage, factor 4 runs 2.9× faster. Accuracy degrades:
- blade f4 forward: core precision 0.22, rigid error 0.84 m;
- storage f4 forward: rigid error 2.3 m;
- storage f4 backward: rigid error 1 mm, the same as factor 1.

No factor is "optimal". Factor 1 is the reference.

### What fails and why (evidence, no tuning)

1. **Free capture range, about 1–2 px per frame.** The free filter is a local IEKF on piecewise-bilinear measured images.
   - On a static textured frame, convergence from a 2.5 px offset works for isolated points.
   - The blade moves about 7 px per frame, storage about 2 px and the laptop lid 3–5 px; v starts at 0. So measured moving points lag (storage 0.15 m of 0.53 m; blade 0.36 m of 0.835 m).
   - On a textureless blade surface, the information along the slide is about 0 (σ about 21 m). This is genuine unobservability.
2. **Static drift from non-converged points.**
   - Points whose last IEKF iterate is an oscillation state acquire spurious velocity through the position–velocity covariance, and the constant-velocity prediction accumulates it: blade static median 0.12 m by frame 59.
   - Points that met the step criterion stay within 1.1 mm.
   - Rejecting non-converged corrections (diagnostic) reduced the drift but removed the remaining moving tracking. Not adopted; reported.
3. **Rigid covariance about 40× overconfident.** The 12D update treats ~20k core point-camera rows as independent measurements, which is the contract's "conditional per-row approximation". On storage backward the posterior σ is 0.043 mm, against a true error of 1.6 mm. Every χ² adequacy test is therefore rejected, and the joint is UNIDENTIFIABLE even when the axis is within 0.1°. On calibrated synthetic sequences the same decision logic is correct (Gate G).
4. **Static false joint.** The pinned extractor resolves a core on static noise, and the overconfident rigid covariance then rejects identity. Gate G fails.
5. **Rigid divergence** (storage forward after frame 43, laptop backward). Accepted rows collapse (14k → 224), NIS per dof rises to 109, and the pose diverges. The diverged pose is still reported.
6. **Noise is uncalibrated.** The free filter's NIS per dof has a median of about 0.02–0.05, so the declared R is far larger than the image noise of these synthetic data.

## 6. Debug pages

`<run>/debug/index.html` links:
- `free_{forward,backward}_3d.html`: canonical, predicted and posterior points; cameras; velocity; traced trajectories; 2σ covariance glyphs; validity layers; exact counts.
- `free_*_rgbd.html`: the recorded RGB, metric depth and mask with initial, predicted and posterior projections, correction arrows, sampled values and innovations; zoom; click to open the 3D view.
- `coherence_*.html`: the 192-node graph, strain over time, seed / recruited / static / ambiguous nodes, candidates, raw vs supported.
- `rigid_{forward,backward}_3d.html` and `rigid_rgbd.html`.
- `joint_*.html` and `bidirectional.html`.
- `numerical_diagnostics.html`.

Numeric logs per pass: `numeric/frame_XXXX.json`, `trace_frame_XXXX.json` (per-point r, H, R, A, b, δ, K, P⁻/P⁺, Joseph check, finite-difference H; a rigid 12D trace) and `timeline.csv`. QA screenshots are in `debug/qa/`; inspected examples are `blade_f1/debug/qa/free_forward_rgbd_f0005_c0.png`, `rigid_rgbd.png`, `coherence_forward.png` and `numerical_diagnostics.png`.

## 7. Implemented / Tested / Passing / Failing / Not implemented

| Item | Implemented | Tested | Passing |
|---|---|---|---|
| Measured sampling, H, R, validity, visibility | yes | Gate A | yes |
| Free 6D IEKF (F1/B1), CUDA batched, Joseph-equivalent posterior | yes | Gate B/C | yes on fixtures; **limited capture range on real data** |
| Subsampling (factor, spatial-stratified, seeded-uniform, boundary) | yes | Gate D | yes |
| Coherence on projective tracks (two weightings) | yes | parity + E2E | implementation yes; outcome scene-dependent |
| Rigid 12D Lie IEKF (F2/B2) | yes | Gate F | yes on fixtures; **diverges on 2 of 6 real passes** |
| Joint reduction, decision, `(q, q̇)` IEKF, F/B comparison | yes | Gate G | logic yes; **decision fails on real data (covariance)** |
| Support from measured projective consistency | yes | E2E fixture | yes on fixture; not reached on real data (no accepted joint) |
| Static scene → no joint | — | Gate G | **FAILS** |
| Debug pages, QA, GT evaluation, profiling | yes | Gate H | yes |
| Triton / custom CUDA kernels | **no** | — | — |
| Image-coordinate uncertainty, colour–depth correlated noise, interpolation-correlated rows | **no** (documented approximations) | — | — |
| `--stage S --resume-from RUN` stage reuse | yes | blade 6-frame run resumed at coherence: free stages symlinked, coherence summary identical, rigid poses bitwise identical | yes |
