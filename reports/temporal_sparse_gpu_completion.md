# Exact temporal RGB-D point IEKF — completion report (2026-11-02)

Contract: `CODEX_RGBDiekf_SURGICAL_EXACT_TEMPORAL_RGBD_IEKF_COMPLETION_2026-11-01.md`.

**Status: `FULL_SPARSE_GPU_NOT_ACCEPTED`.** G4 is additionally blocked by a measured exact-Schur `FULL_RESOURCE_LIMIT` (section 10).
- **Passed:** G0, G1, G2 and G3. Full-cloud USB and Blade frames 0–2 on all 8 cameras were estimated, and their complete joint covariance was verified.
- **Not run:** G4 (lag slide, backward pass), G5 (GT scientific validity) and G6 (scaling).
- **Inherited auditor gate:** passes for USB and fails for Blade on its own float64 oracle (section 4).

The model is unchanged. No graph, prior, threshold or approximation was added.

## 1. Local tree and changes

- **Baseline:** HEAD `f77492dc0c8fe3cf5f28820ba5fe1f8ed959cb58`, plus an uncommitted earlier attempt (symmetric cuDSS mode, a float64 selected-inverse draft, a capture/replay tool). That attempt was used as the starting point. Nothing is committed.
- **Untouched:** the legacy estimator, simplified mode and the temporal row operator.

| File | Change | Defect / feature it addresses |
|---|---|---|
| `estimator/temporal_cudss.py` | Production factorization is now symmetric-indefinite: nested dissection + Bunch–Kaufman, zero pivot epsilon. Inertia certificate. Empty the Torch cache before native handles. Multithreaded reordering layer. Diagnostic-only `max_lu_nnz` override. | S2 `INFO=-1`; native allocation failure under the Torch cache; analysis time 11.3 s → 5.0 s |
| `estimator/temporal_sparse_solve.py` | `ReducedLinearizedSystem` split out with identical arithmetic. Expected KKT inertia. `all_point_marginals_selected_inverse_gpu` with a certificate, an exact host-offload memory schedule and a failure archive. `offload_device_tensors`. | All-point covariance was infeasible: 204,660 RHS ≈ 4,900 s per frame |
| `estimator/temporal_selected_inverse_gpu.py`, `cuda/temporal_selected_symbolic.cpp` | Rewritten as a double-double selected inverse (Takahashi on an unpivoted LDLᵀ), in place, chunked. CHOLMOD METIS ordering of K plus the query cliques. | float64 instability; FMA contraction; register spill; memory |
| `estimator/temporal_sparse_nonlinear.py`, `estimator/temporal_sparse_run.py` | Final-linearization hash; covariance must be queried at it. Runner uses the selected inverse and parks its own state during inversion. | Provenance (§4.3) |
| `tools/run_temporal_rgbd_critical.py` | `--pcd-subsample-method`, `--pcd-subsample-seed` forwarded to every child. | §7 |
| `tools/replay_temporal_sparse_failure.py` | Captures S0 and S2 plus key maps; `analyze` stage. | §3 |
| `tools/verify_temporal_full_frames.py`, `tools/arbitrate_component_oracles_mp.py` | Read-only G3 verifier; 50-digit oracle arbiter. | Evidence |
| `debug/temporal_debug.py` | Integer dtype for debug IDs when no traced ID survives subsampling. | Crash at factor > 1 |
| `train.py`, `tools/temporal_batch_runtime.py`, `tools/test_temporal_rgbd.py`, `tools/report_temporal_sparse_gpu.py` | New sources hashed, new test modules registered; the auditor skips `.cpp` in its Python line map. | Integration |
| `tests/test_temporal_surgical_completion.py` (new), `tests/test_temporal_selected_inverse_gpu.py` | 14 new tests. | G0–G2 |

## 2. MODEL_PARITY (equations → code → independent witness)

| Equation (contract §2) | Implementation | Independent witness |
|---|---|---|
| Warp prediction, `H_t`, `H_r`, `+w_a I4` reference pixels; zero velocity columns | `temporal_rgbd_terms.temporal_prediction`, `TemporalSoAGraph.evaluate_factor_residuals_and_blocks_gpu` (unchanged) | Existing central-difference / dense-oracle tests (75/75 pass) |
| Raw pixel likelihood included once; injective `(camera, frame, pixel)` key | `UniquePixelTableGPU` | Real S0/S2: 184,613 unique = registered keys; round-trip injective; 0 cross-camera collisions; 0 orphans; RAW_PIXEL factors = active keys |
| Hard ideal-warp equality; FEJ Schur | Pivot substitution (unchanged) | Equality residual ≤ 2e-11 certified at every solve |
| Constrained MAP of the linearized problem | Same augmented KKT, symmetric-indefinite cuDSS | Refined residual certificate; KKT inertia = (m+p, n) on every real factorization |
| Joint posterior `P_i = Q_i Cov Q_iᵀ` for all points | Takahashi selected inverse of the SAME KKT in double-double | Refined cuDSS solves (≤ 9.5e-11 correlation units); 80-bit refined references; exact rational test; all-block dual-ordering agreement ≤ 7e-18 on Blade; dense constrained inverse in the tests |
| Connectivity = actual nonzero coefficients only | `connected_columns_gpu`; line-search ties never enter `information` | GPU = independent CPU union-find on real S0 and S2; tie test |

## 3. S2_FAILURE_CAUSE

Diagnostic: `tools/replay_temporal_sparse_failure.py capture|analyze`. Evidence is in `results_11_10_26/RGBDekf/temporal_S0_S2_diagnosis_20261102/analysis/analysis.json`, captured from the unchanged full-cloud USB input. The capture reproduces the recorded S1 acceptance: objective 44062961178.87926 → 43743829644.68693, 30 backtracks.

**What `INFO=-1` was.** Production used general BTF+COLAMD LU. That method only learns its fill during factorization, so it is bounded by `MAX_LU_NNZ`. The bound was derived from free VRAM (55.6 M entries; BTF's own peak-device estimate 6.39 GB). NVIDIA documents a device runtime error in `CUDSS_DATA_INFO` when that bound is insufficient. Controlled reproduction on the real S0:
- default bound: factorizes (LU nnz 7.6 M);
- bound at half, or exactly equal to, its measured fill: `INFO=-1`.

S2 needs more than 55.6 M, and a larger bound cannot fit 8 GB. It is neither singularity nor a Torch OOM. The symmetric-indefinite nested-dissection factorization of the identical S2 operator succeeds: 36.2 M factor entries, 0.72 GB device peak, inertia exactly [354584, 1439505].

**Why S2 is so much more coupled than S0:**

| | S0 | S2 |
|---|---|---|
| Largest true component (scalars) | 403 | 371,166 |
| Bilinear weights exactly zero | 265,938 / 354,584 | 164,759 |
| Raw keys shared as nonzero corner by ≥ 2 points | 0 | 37,075 |
| Largest point component | 68 | 26,764 |

- **Two sources of the change:**
  - 24,644 warp rows moved to a new reference cell;
  - 27,307 corners on unchanged cells went from exactly 0 to nonzero.
- **Cell changes alone already give a 25,847-point component.**
- **Every coupling is accounted for:** every actual nonzero of the reduced operator is traced to a predicted analytical factor (0 untraced pairs, S0 and S2).
  - Predicted-but-absent pairs (10,060 at S0, 7,855 at S2) are all raw-pixel nodes.
  - Corner coefficients are −w ≠ 0, so the only predicted raw-pixel node that can vanish is the row's own source pixel. Its coefficient cancels exactly when the source enters x_t and x_r equally.
- **Nothing was removed:** no duplicate likelihoods and no stale keys, so no bug coupling had to be removed.

## 4. GPU_SOLVER_CERTIFICATE

**MAP solve:** refined cuDSS solves, componentwise backward error ≤ 2e-12, hard-equality residual ≤ 2e-11, inertia matching the proper-posterior prediction on every factorization.

**Covariance.** The original colored-RHS method was exact but needed 204,660 solves per USB frame (≈ 4,900 s). It is replaced by selected inversion of the same KKT. Along the way the certificate exposed three numerical defects, each fixed and covered by a test:

1. **Unpivoted float64 LDLᵀ is unstable.** Growth is about ‖A‖², with whitened Jacobians around 1e6.
   - Errors reached 5.45e-6 correlation units on Blade against an extended-precision refined arbiter (cuDSS: 3.5e-10).
   - Two independent orderings disagreed by up to 0.17 (Blade) and 4.6e-5 (USB S2).
   - A measurement-first ordering made it worse (2.1e-5).
   - Fix: the same recurrences in double-double arithmetic.
2. **FMA contraction silently cancelled the double-double error terms.** The result was no better than float64 (2.9e-9 vs 1.0e-9 on a 900×900 test).
   - Fix: every double-double kernel is launched with `enable_fp_fusion=False`.
   - After the fix: 4.4e-17 vs an 80-bit refined truth; Blade sampled blocks 9.9e-17; all 34,369 Blade blocks agree across two orderings to ≤ 6.9e-18.
3. **Kernels whose block width scaled with the longest row spilled registers** and failed to launch on a 1/8 subset.
   - Fix: fixed-width chunked walks (≤ 512), still at double-double accuracy (test with chunk width 8).

**Per-frame certificate (final runs).** Each frame compares 32 blocks against refined cuDSS solves of the original K, including blocks inside the giant component; tolerance √eps = 1.5e-8 correlation units.

| Run | Frame | Max error (corr. units) | Giant component (scalars) |
|---|---|---|---|
| USB | 1 | 2.5e-12 | 371,135 |
| USB | 2 | 9.5e-11 | 541,234 |
| Blade | 1 | 3.5e-10 | 27,279 |
| Blade | 2 | 5.7e-10 | 49,020 |

The Blade values equal the cuDSS reference's own measured error, so the selected inverse is the more accurate side.

**Inherited auditor** (`tools/report_temporal_sparse_gpu.py`, saved in `reports/temporal_sparse_gpu_completion_evidence/`):
- USB passes the three-frame gate (4/4 dense real-component oracles, SPD covariance).
- Blade fails it. Its rule requires QR condition < 1e6; Blade's saved components have 1.7–3.9e6 (tiny measured depth noise).
- The 50-digit arbiter (`component_oracle_50digit_arbiter.json` in each run) shows the production step is accurate to 1.9e-15–2.9e-14 and the covariance to ≤ 2.5e-15. The auditor's own float64 QR is off by 2.2e-12–2.6e-11.
- I did not change the auditor's acceptance rule. The automatic >3-frame gate therefore stays locked, and `reports/temporal_sparse_gpu_validation.json` was not overwritten.

## 5. ACTUAL_FRAME_EVIDENCE (final code, factor 1, all 8 cameras, forward, ideal warp)

Verified with `tools/verify_temporal_full_frames.py`, written as `G3_frame_verification.json` in each run. Every frame has: journal digests matching, finite values, every point's 6×6 marginal SPD, predictions equal to x⁺ + dt·v⁺ (≤ 6e-8 m at float32 storage), nonzero corrections, covariance hash equal to the final accepted linearization, and static-LDL inertia = cuDSS inertia = expected. Uncommitted slots remain NaN.

| Scene (points) | Run | Frame | Rows | Points assimilated | IEKF objective | Frame s |
|---|---|---|---|---|---|---|
| USB_100109 (95,967) | `temporal_rgbd_20261102_011901_75a43156` | 1 | 88,646 | 58,487 | 4.4062961e10 → 4.3743616e10 | 182 |
| | | 2 | 83,187 | 34,284 | 1.1119344e11 → 1.1117314e11 | 657 |
| blade_103706 (34,369) | `temporal_rgbd_20261102_011651_b9364f61` | 1 | 30,083 | 20,733 | 1.3276e14 → 1.2766e14 | 41 |
| | | 2 | 29,668 | 18,738 | 3.1560e14 → 3.1385e14 | 79 |

- **Per-frame outputs:** `posterior_*.npy` and `posterior_covariance.npy` (frame-indexed, journal `completed_frames.json`), `temporal/frame_000k.json` (rows, reference lifecycle, raw residual RMS, solver history), `temporal/sparse_profile_000k.json`, `temporal/traced_points/*.json`.
- **Noise provenance:** the measured sensor R comes from the scene's `noise_calibration.json` via the run manifest. The warp model is `IDEAL_WARP_ASSUMPTION_UNCALIBRATED`, which is not a calibration.
- **Convergence flag:** `converged=false` on every frame. All 10 iterations were accepted, but 54–130 true components ended frozen at their last valid common-domain state after 25 rejected halvings.

## 6. DOWNSAMPLING_VALIDATION

- **Forwarding:** method and seed reach every child of every mode (dry run and test).
- **Sampler:** the existing seeded-uniform sampler gives a bitwise identity at factor 1; at factors 2/4/8 it is deterministic per seed, differs across seeds, has no replacement, keeps original IDs sorted, and has size ⌈N/f⌉ (test).
- **USB factor 8, seed 0, all cameras, same full solver:** `OS_HARD_TIMEOUT` at 900 s during the frame-1 exact covariance (run `temporal_rgbd_20261102_005128_e80cc47b`).
- **Why the subset is harder:** its KKT (n = 663 k) has a selected-inverse factor of 167 M entries (252 per variable vs 31 at factor 1), a longest row of 23,127 and 15,296 levels. Removing occluders changes first-hit visibility and therefore the coupling structure.
- **Consequence:** the downsampled cloud is *not* a cheaper proxy and cannot be compared with factor 1 as an unchanged inference. An earlier factor-8 attempt crashed in the debug viewer (fixed) and then hit the register-spill launch failure (fixed by chunking).

## 7. RESOURCE_PROFILE (RTX 4070 Laptop, 8 GB)

| Frame | KKT rows / nnz | cuDSS factor nnz, analysis | Selected-inverse factor nnz | Covariance s (symbolic / LDL / inverse) | Peak Torch alloc |
|---|---|---|---|---|---|
| USB 1 | 1.79 M / 4.48 M (lower) | 36.4 M, 5.0 s | 57.3 M | 81 (13 / 39 / 24) | 3.30 GB |
| USB 2 | 2.72 M / 12.1 M | 160.3 M, 10.2 s | 198.8 M | 450 (33 / 233 / 160) | 7.12 GB |
| Blade 1 | 0.64 M / 1.19 M | 1.9 M, 1.1 s | 5.4 M | 7 | 1.01 GB |
| Blade 2 | 0.98 M / 2.54 M | 5.2 M, 2.1 s | 12.0 M | 14 | 1.53 GB |

- **USB frame 2 fits 8 GB only through the exact memory schedule:** cuDSS factor released, 1.18 GB of operators plus 65 caller tensors parked on the host, restored bit-identically afterwards, then the same K refactorized.
- **The double-double workspace is 28 B per factor entry.** A frame-3 window larger than about 6 GB of workspace would stop with `FULL_RESOURCE_LIMIT` and report its measured bytes.
- **The 120 s child budget is infeasible** for the exact full-cloud model on this GPU: USB needs about 860 s for three frames and Blade about 130 s. The commands below therefore use measured, explicit hard deadlines. Nothing in the model was simplified to meet a budget.

## 8. REGRESSION_REPORT

- **Focused suite:** `python -B tools/test_temporal_rgbd.py` → 75/75 pass (61 baseline + 14 new), no source changed during the tests (`tests/temporal_test_results.json`).
- **Not re-run:** the broader legacy discovery suite, because its numerics are untouched.

## 9. EXACT_USER_COMMANDS

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
# G3, full cloud (measured budgets; ~860 s USB, ~130 s Blade)
bash tools/run_temporal_rgbd_critical.sh --mode full --direction forward --frame-limit 3 --scenes USB_100109 \
  --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --stop-after free --temporal-runtime-budget-seconds 2400
bash tools/run_temporal_rgbd_critical.sh --mode full --direction forward --frame-limit 3 --scenes blade_103706 \
  --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --stop-after free --temporal-runtime-budget-seconds 1800
python -B tools/verify_temporal_full_frames.py RUN/<scene>_full/free_forward --output RUN/G3_frame_verification.json
python -B tools/arbitrate_component_oracles_mp.py RUN/<scene>_full/free_forward/temporal/oracle_frame_*.npz --output RUN/arbiter.json
# S0/S1/S2 diagnosis from a finished full-cloud USB run
python -B tools/replay_temporal_sparse_failure.py capture --cached-run RUN/USB_100109_full/free_forward --output OUT/capture
python -B tools/replay_temporal_sparse_failure.py analyze --linearization OUT/capture --output OUT/analysis
```

## 10. G4 lag-boundary elimination: measured FULL_RESOURCE_LIMIT (exact Schur fill)

The spec's G4 uses 4 frames with lag 3, which is behind the locked large-run gate. The same code path (`marginalize_lag_boundary_sparse_gpu`, exact sparse FEJ Schur of the anchor charts and frame-0 raw pixels) was exercised within the permitted 3 frames by using lag 2: frame 0 expires before the frame-2 solve. In both scenes frame 1 was published normally. Frame 2 stopped with `FULL_RESOURCE_LIMIT`, and no invalid frame was published.

| Scene | Run | Induced Schur entries | Required bytes | Eliminated components with boundary | Largest boundary (rows) | Boundary median / p99 |
|---|---|---|---|---|---|---|
| blade_103706 | `temporal_rgbd_20261102_014147_3703b4dd` | 975,727,216 | 31.3 GB | 79,882 | 30,745 | 1 / 35 |
| USB_100109 | `temporal_rgbd_20261102_014254_f5034d7c` | 163,651,120,150 | 5.24 TB | 144,043 | 404,511 | 1 / 35 |

- **The fill is mathematical, not an artifact.** Eliminating a connected set of expired variables makes its whole boundary mutually dependent, so the information is dense per component (Σ boundary²).
- **One component dominates.** It is the giant shared-reference-pixel component described in section 3.
- **The explicit dense FEJ marginal prior therefore cannot be formed at full cloud** on 8 GB or on any available GPU (USB: terabytes).
- **Classification fix:** this condition was previously raised as a generic numerical failure. It now raises `SparseResourceLimit`, recording the boundary statistics.

**Unimplemented, mathematically identical alternative (needs a theoretical decision, not started):** keep the expired variables' FEJ-linearized rows inside the sparse system and eliminate them inside the solve (an implicit Schur complement), instead of forming the dense marginal information.
- The posterior is the same.
- Memory stays sparse, but the linear system keeps the frozen rows, so window cost no longer stays bounded by the lag. Whether and when frozen blocks may later be eliminated is a design choice the contract reserves.
- A falsification test would be: implicit vs dense Schur on small complete components (existing QR oracles), then the lag-2 Blade run.

## 11. Open items (why not ACCEPTED)

- **G4** is blocked by the exact-Schur resource limit above. The backward pass and camera handover were not run. The 4-frame lag-3 form is also behind the large-run gate, which the auditor's Blade oracle rule keeps locked (section 4).
- **G5** (GT falsification of the local-translation warp) and **G6** (scaling) were not run.
- **Factor-8 USB** does not finish within 900 s (section 6).
- **Speed:** USB frame-2 covariance takes 450 s, dominated by the double-double LDL and inverse.
