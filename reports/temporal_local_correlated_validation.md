# Local-correlated temporal RGB-D IEKF: implementation and bounded validation (2026-11-02)

**Status:**

- **Implementation:** implemented, numerically verified, and complete for full-cloud frames 0–2 on USB and Blade (all 8 cameras).
- **Physical motion:** `NOT_RECOVERED`. Moving points are not tracked by local-correlated, simplified, or globally correlated full.
- **Warp model:** the GT-motion warp check flags `TEMPORAL_WARP_MODEL_INVALID_OR_BIASED` for moving points (depth channel) under `--temporal-warp-noise ideal`.
- **Later gates:** Gates 3 and 4 (11 frames) were not run. Per spec section 8 the work stops for review, because the physical warp is inaccurate even at GT.

Spec: `CODEX_RGBDiekf_LOCAL_CORRELATED_TEMPORAL_IEKF_IMPLEMENTATION_2026-11-02.md`. Baseline: `9465cb1`. All changes are uncommitted (section 1). Machine-readable companion: `reports/temporal_local_correlated_validation.json`.

## 0. Assumption statement (stated in every manifest)

`temporal_mode=local-correlated`, `cross_point_covariance=block_diagonal_assumed_density`, qualifier
`BLOCK_DIAGONAL_ACROSS_POINTS; TEMPORAL_AND_SENSOR_CROSS_CORRELATION_RETAINED_WITHIN_POINT`.

The posterior is the point-local Gaussian linearization. It is conditional on the hard first-hit owners, the camera-local reference policy and the active supports. Two points that use the same raw pixel carry independent local copies of its sensor error. It is not the globally correlated `full` posterior, not globally exact nonlinear inference, and not smoothing.

`--temporal-warp-noise ideal` sets only `R_warp = 0`, an explicit uncalibrated diagnostic assumption. Raw sensor errors stay active as local variables.

## 1. Working tree

Base `9465cb109f22e4047d7132b3f9d2f22f3f5d633e`. Nothing is committed.

**New files**

| File | Purpose |
|---|---|
| `estimator/temporal_local_covariance.py` | Packed variable-length local Gaussians; source chart; prediction; retention and expiry; certificates. |
| `estimator/temporal_local_iekf.py` | Stacked point-local residual and Jacobian; fixed-prior IEKF; Joseph posterior; exact local marginalization. |
| `estimator/temporal_local_run.py` | `LocalFilter` and the F1/B1 runner (journal, viewer, traces). |
| `tests/test_temporal_local_{covariance,iekf,pipeline,motion}.py` | 31 tests. |
| `tools/evaluate_temporal_local_pilot.py` | Post-hoc GT evaluation only. |

**Modified files**

| File | Change |
|---|---|
| `train.py` | `--temporal-mode local-correlated`; `--temporal-local-chunk-mib` (default 512; batching only); dispatch; `STAGE_SOURCES`. |
| `tools/run_temporal_rgbd_critical.py` | `--mode local-correlated`; chunk budget forwarded only to local children. |
| `tools/test_temporal_rgbd.py` | Registers the 4 new modules and their witnesses. |
| `debug/temporal_debug.py` | Point-local covariance-retention panel and accepted/prediction-only status in the live viewer. |
| `debug/projective_debug.py` | Pre-existing bug fix: `np.union1d` of empty traced ID lists gave float indices. |
| `tools/run_status.py` | Temporal-batch support, read from the journals. Not a pipeline source. |
| `tests/temporal_test_results.json` | Regenerated. |

`full`, `simplified` and `legacy` code paths are unchanged. `temporal_simplified_iekf.py` is only imported, for `point_domains` and `outside_points`.

Source SHA-256 prefixes, identical in every pilot `run_manifest.json`:

| File | SHA-256 prefix |
|---|---|
| `temporal_local_covariance.py` | `fe3942780e1fb149` |
| `temporal_local_iekf.py` | `3849baddd3f93c69` |
| `temporal_local_run.py` | `6d3f32370d31ab0e` |

## 2. Local state and equation-to-code map

Per point `i`, the columns of its Gaussian are, in order:

1. `s_{i,t}` (6, the current state; always the first block).
2. Every reference state that a camera can still use, at processing-order age < lag. While the anchor frame is referenced, this block is instead the 5D source chart `θ = (δu, δv, v)`.
3. Every raw sensor error `η_{i,(camera,frame,pixel)}` (4) that this point has already used and can still reuse.

Fresh keys touched by the current linearization are appended with one `N(0, R_sensor)` prior. Storage is a block table plus a flat mean (Σnᵢ) plus a flat covariance (Σnᵢ²). There is no padded all-point cube.

| Spec equation | Function | Independent witness |
|---|---|---|
| 1.1 prediction, Cov(s_t,·) = F·Cov(s_{t−1},·), Q once | `predict_local`, `current_jacobian` | dense affine transform; chart chain rule (`PredictionTests`, 2 tests) |
| 1.2 one prior per (camera, frame, pixel) key, reused within a point | `LocalFrameUpdate.linearize`, `Extras`, `BlockIndex` | one variable for the source key that enters the chart and a bilinear neighbour, with the cross-camera innovation equal to H Σ Hᵀ from finite differences (`test_one_key_one_variable…`) |
| 1.3 f = Σ w_a (y_a − η_a) + e₄ΔZ + η_t; `R_warp` additional only | `linearize` (clean `y − η` in `temporal_prediction`), `_system` | NumPy model of the spec equation (`Witness.f`) |
| 1.4 H_t, H_ref, H_η_ref = −w_a I, H_η_cur = I; duplicate keys summed | `_system`, `_add_blocks` (accumulating) | 5-point FD: max \|H − FD\| = 3.2e-12 (state reference), 5.9e-12 (chart reference); velocity columns exactly 0 |
| 1.5 fixed-prior IEKF step | `proposal`, `solve` | QR (R_w > 0) and null-space QR (R_w = 0) of the same linearized MAP: agreement 1e-10 (8 subtests, CPU and CUDA) |
| 1.5 Joseph posterior once at the accepted iterate | `posterior` | information form, algebraic (I−KH)Σ, constrained projection: agreement 1e-9 (relative to max); sequential update equals stacked update |
| 1.6 minimal source chart; source depth correlation | `SourceChart`, `initial_local_gaussian` | direct backprojection 1e-14; FD Jacobians; Cov(x₀, η_D) against 200k Monte Carlo samples (5σ) and the closed form; no redundant 6D x₀ |
| 1.7 retention and exact local marginalization | `retention_mask`, `retain`, `record_expiry`, `check_no_recreation` | slice equals square-root Schur of the information (3 drop patterns); lag 2 and lag 3 rules; expired key re-creation raises |
| 1.8 global visibility, local covariance | `solve` (common domain, mask feasibility, per-point MAP descent, owner rollback; reused rules) | rejected iterates add no information; no-row points keep the exact prediction (bitwise) |
| Cross-point omission (spec test 11) | design: per-point keys | global dense oracle P_ij max 1.3e-6 ≠ 0; local copies equal the per-point oracle |

**Acceptance objective.** The objective is the exact MAP objective of the local model with the fresh current errors profiled out:

`J = ½ δᵀ z + ½ Σ eᵀ (R_cur + R_warp)⁻¹ e`, with `δ = Σ z`.

It is finite for `R_warp = 0` and needs no Σ⁻¹. Acceptance otherwise uses the existing simplified rules unchanged.

## 3. Tests (Gate 0)

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
~/miniconda3/envs/rigeuler3dgs/bin/python -B -m unittest tests.test_temporal_local_covariance tests.test_temporal_local_iekf tests.test_temporal_local_pipeline tests.test_temporal_local_motion -v
~/miniconda3/envs/rigeuler3dgs/bin/python -B tools/test_temporal_rgbd.py
```

- **Result:** 31/31 new tests pass. The focused suite passes 106/106 (the 75 existing plus 31 new) with 0 failures, 0 errors, 0 skips, CUDA available, and no sources changed during the tests. Evidence is in `tests/temporal_test_results.json` (`local_correlated_witnesses`).
- **CPU/CUDA parity on the tiny pipeline:** posterior covariance agrees to 1.1e-16.
- **Controlled motion across an interpolation-cell boundary, ±0.1 m:** relative displacement is recovered to 0.32 mm, velocity 2.492 m/s against 2.500 m/s.
- **Static fixture:** zero drift.
- **Persistence:** Cov(state, retained raw error) = 2.8e-6 survives propagation and is reused at the next frame.
- **Local vs simplified on the same controlled example:** the frame-2 covariance differs by 531% relative, and the position by 0.05 mm.

## 4. Bounded pilots: publication evidence

Both batches ran with frozen code, `--temporal-warp-noise ideal --temporal-lag-frames 3`, frames 0–2, seeded-uniform seed 0. All runs `COMPLETE`. Every frame is journal-verified, so `completed_frames.json` and `frame_valid.npy` are consistent.

The two batches are:

- **Gate 1:** `temporal_rgbd_20261102_085707_71acfdc9`, uniform factor 8, both directions. B1 starts at the endpoint of the truncated 3-frame prefix (frame 2), not frame 59.
- **Gate 2:** `temporal_rgbd_20261102_090025_a7b905eb`, factor 1 (every original point), forward.

Both live under `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/`.

**Gate 2, full cloud, all 8 cameras:**

| Scene / mode | Points | Frame | Rows | Observed | Accepted | s/frame | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| USB local-correlated | 95,967 | 1 | 88,646 | 61,539 | 53,259 | 30.3 | 2.30 GB |
| | | 2 | 81,484 | 58,218 | 50,138 | 27.7 | 3.42 GB |
| USB simplified | 95,967 | 1 | 88,646 | 61,539 | 50,255 | 12.7 | — |
| | | 2 | 83,251 | 58,745 | 48,226 | 11.8 | — |
| Blade local-correlated | 34,369 | 1 | 30,083 | 21,252 | 20,462 | 13.0 | 1.35 GB |
| | | 2 | 29,713 | 21,075 | 20,413 | 13.3 | 1.85 GB |
| Blade simplified | 34,369 | 1 | 30,083 | 21,252 | 20,329 | 5.8 | — |
| | | 2 | 29,799 | 21,097 | 20,340 | 6.3 | — |

**Local covariance validity at full cloud (USB f1 / f2; Blade f1 / f2).** Every frame passed the runtime certificates: finite values, symmetric, PSD in correlation units ≥ −1e-9, Cholesky-SPD 6×6 marginal for every point, and 0 forbidden key re-creations.

| Certificate | USB f1 / f2 | Blade f1 / f2 |
|---|---|---|
| Min local correlation eigenvalue | −2.0e-16 / 1.6e-8 | 2.9e-16 / 4.0e-14 |
| Min innovation eigenvalue | 9.2e-10 / 9.1e-10 | 6.5e-15 / 7.1e-15 |
| Joseph vs algebraic gap (correlation units) | 7.2e-9 / 2.9e-9 | 2.2e-4 / 1.2e-5 |

The Blade Joseph/algebraic gap needs qualifying. It occurs where S is extremely ill-conditioned, because Blade's measured sensor variance is about 1e-14. The algebraic (I−KH)Σ form is the unstable one there, and the published covariance is Joseph. An extended-precision arbiter on these real points was **not run** (open item).

## 5. Memory, runtime and local size distribution (Gate 2)

**Retained local dimension after expiry** (min / median / p95 / p99 / max):

| | Frame 1 | Frame 2 |
|---|---|---|
| USB | 6 / 15 / 23 / 23 / 31 | 6 / 10 / 18 / 22 / 28 |
| Blade | 6 / 15 / 23 / 23 / 35 | 6 / 10 / 14 / 18 / 26 |

- **Raw keys per point:** 0 / 1 / 2–3 / 3 / 5–6.
- **Packed bank:** USB 207.9 MB (f1) and 98.8 MB (f2); Blade 76.0 MB and 30.8 MB. A padded cube would need N·max²·8 B.
- **Largest in-update working block:** 167–173 (USB) and 119–125 (Blade).
- **Expired raw keys per frame:** USB 310,686 and 349,988. Within-point key reuse rows: USB 27,492 and 70,122. Source-key reuse rows: 27,492 and 4,289.

**USB f1 stage times:**

| Stage | Seconds |
|---|---:|
| Nonlinear solve | 26.6 |
| … of which ownership and row sampling (z-buffer and rows; 288 linearizations) | 11.0 |
| Covariance and marginalization | 2.1 |
| Viewer | 1.1 |
| Predict | 0.1 |

**Per-stage GPU peaks:**

| Run | Predict | Solve | Covariance |
|---|---:|---:|---:|
| USB full cloud, f1 | 1.10 GB | 2.30 GB | 2.28 GB |
| USB full cloud, f2 | 2.50 GB | 3.42 GB | 3.34 GB |
| USB factor 8, f1 | 0.52 GB | 0.98 GB | 1.10 GB | Local mode costs about 2.3× simplified per frame. On the same USB frames 0–2, the exact `full` mode took 182 s and 657 s.

## 6. Post-hoc GT accuracy on matched points (identical IDs, sensor data and 6×6 frame-0 marginal prior)

The local chart-implied frame-0 marginal equals simplified's canonical Pxx to 7e-17. The three-way file is `temporal_rgbd_20261102_090025_a7b905eb/posthoc_gt_full_local_simplified.json`. The `full` runs are the earlier verified runs `temporal_rgbd_20261102_011901_75a43156` (USB) and `…011651_b9364f61` (Blade), with identical settings.

**Full cloud, forward.** Position error in mm (median / p90 / p99). GT moving points: USB 43,036 of 95,967; Blade 3,218 of 34,369. True mean speed: USB 0.863 m/s, Blade 0.835 m/s.

| Scene | Mode | Static f1 | Moving f1 | Moving f2 | Moving velocity error f2 (m/s) | Accepted / observed f1 |
|---|---|---|---|---|---:|---:|
| USB | local-correlated | 0 / 0 / 0 | 15.70 / 22.26 / 22.91 | 31.27 / 44.39 / 46.26 | 0.922 | 0.865 |
| USB | simplified | 0 / 0 / 0 | 15.77 / 22.27 / 22.63 | 31.44 / 44.45 / 45.52 | 0.927 | 0.817 |
| USB | full | 0 / 0 / 0 | 15.78 / 22.26 / 22.75 | 31.56 / 44.53 / 45.51 | 0.931 | 0.950 |
| Blade | local-correlated | 0 / 0 / 0 | 14.16 / 14.16 / 16.40 | 28.31 / 28.34 / 33.76 | 0.835 | 0.963 |
| Blade | simplified | 0 / 0 / 0 | 14.16 / 14.16 / 16.16 | 28.31 / 28.32 / 32.54 | 0.835 | 0.957 |
| Blade | full | 0 / 0 / 0 | 14.16 / 14.16 / 15.74 | 28.31 / 28.31 / 31.48 | 0.835 | 0.976 |

Moving-point error equals the true displacement, and velocity error equals the true speed, in **all three** covariance models. Static points are exact to p99 at full cloud.

**Moving-point NEES (median, full 3×3 position covariance), f1 / f2:**

| Scene | local | full |
|---|---|---|
| USB | 1.34 / 1.64 | 1.72 / 4.68 |
| Blade | 0.69 / 0.70 | 0.70 / 0.70 |

These covariances are large because no motion was learned. NIS and NEES are conditional on each mode's own approximation, and samples are correlated.

**Factor 8 (Gate 1).** The moving-point result is identical, in both directions. In addition, local moves static points more than simplified does. USB static error p90 is 0.93 mm against 0.40 mm at f1, and 2.50 mm against 1.32 mm at f2. Local is overconfident on moving points at f2, with moving NEES median 384 (USB) and 162 (Blade).

**Why moving points do not move.** Three measured facts point to the acceptance rule.

1. **True image motion.** The GT image motion of moving points is a median of 1.28 px per frame (USB) and 1.45 px (Blade), with p90 2.6 px and 2.0 px. Only 40% (USB) and 25% (Blade) move by at most 1 px.
2. **Rejection reasons.** At the full IEKF step, rejections are almost all `common_domain_changed`, the inherited fixed current owner-pixel domain.

   | Frame (full cloud) | Domain changed | Mask | No MAP descent |
   |---|---:|---:|---:|
   | USB f1 | 18,639 | 44 | 245 |
   | Blade f1 | 1,579 | 7 | 58 |

3. **Accepted step size.** Accepted moving-point steps have a median of 0.6 mm (USB) and 0.01 mm (Blade), with p90 about 3 mm. The step needed is 15.9 mm (USB) and 14.2 mm (Blade).

The domain rule therefore rejects any update that carries a projection across an integer pixel. Starting from a zero-velocity prior, velocity is never learned.

**GT-motion warp check (spec test 21).** References are the previous frame and GT owners, with the same masks and support rules. Rows are normalized by the sensor-only innovation, which is χ²₄ (median 3.36) if the warp is exact.

| | Static rows | Moving rows |
|---|---|---|
| USB, at GT | NIS 0 (synthetic static pixels identical) | NIS median 8.2 / 6.4 (f1 / f2), p90 689 / 773; RGB standardized residual 0.16; depth 2.0σ |
| Blade, at GT | NIS 0 | NIS median 5.9e4 / 7.9e3; depth 243σ (4–5 mm RMS) against a measured sensor variance of about 1e-14 |
| At the estimated states | NIS 0 | NIS median 1,282 (USB) and 9,600 (Blade) |

The data are thus far better explained by GT motion than by the estimates, so the information exists. However, the depth part of the warp is biased at GT relative to the measured sensor noise. Under the ideal (R_warp = 0) assumption this is `TEMPORAL_WARP_MODEL_INVALID_OR_BIASED`. It was **not** "fixed" by inflating covariance.

**Traced IDs** (full cloud; complete records in `temporal/traced_points/*.json`):

| ID | GT label | Error f1 / f2 (mm) | Speed estimate vs true (m/s) | Notes |
|---|---|---|---|---|
| USB 67827 | static | 0 / 0 | — | |
| USB 85696 | moving | 9.7 / 19.6 | ≤0.005 vs 0.573 | |
| USB 87290 | moving | 15.1 / 30.3 | 0.12 vs 0.94 | f2 `REJECTED_PREDICTION_ONLY` |
| Blade 14104 | static | 0 / 0 | — | |
| Blade 27016 | static (post hoc) | 0.31 / 0.36 | — | |

## 7. Answer to the primary scientific question

**No, not on these pilots.** Retaining each point's two-time reference-state and sensor-error correlations is implemented exactly, verified against independent witnesses, and affordable at full cloud: about 30 s per USB frame, a 3.4 GB peak, and no global Schur. It still does not improve motion recovery over simplified, and neither does the globally correlated `full` model.

All three fail on moving points through the same shared discrete acceptance rule and a GT-biased depth warp, not through the covariance model. On static points local is equal to simplified at full cloud and slightly worse at factor 8.

## 8. Gates

| Gate | Result |
|---|---|
| 0: focused and independent tests | **PASS** (106/106) |
| 1: factor 8, 3 frames, USB and Blade, both directions | **RUN, COMPLETE, numerically valid.** Physical motion: **FAIL** (moving not tracked) |
| 2: factor 1, 3 frames, all 8 cameras | **RUN, COMPLETE, valid local covariance, published frames 0–2.** Physical motion: **FAIL** |
| 3: 11 frames, 4 scenes, factor 4 | **NOT RUN.** Stopping rule: warp inaccurate at GT; Gates 0–2 do not "truly pass" physically |
| 4: backward 11 frames | **NOT RUN** (same reason) |
| 5: 60 frames, all scenes | **NOT AUTHORIZED** |

## 9. Commands

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf"
DATA="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
```

Each launch creates a new `temporal_rgbd_*` folder. Watch it from a second terminal:

```bash
OUT=$(ls -dt "$ROOT"/temporal_rgbd_* | head -1); ~/miniconda3/envs/rigeuler3dgs/bin/python tools/run_status.py "$OUT" --watch 10
```

Gate 1, as run:

```bash
bash tools/run_temporal_rgbd_critical.sh --mode local-correlated,simplified --direction both --frame-limit 3 \
  --scenes USB_100109,blade_103706 --pcd-subsample-factor 8 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 --stop-after free --temporal-runtime-budget-seconds 600
```

Gate 2, as run:

```bash
bash tools/run_temporal_rgbd_critical.sh --mode local-correlated,simplified --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --temporal-lag-frames 3 --stop-after free --temporal-runtime-budget-seconds 900
```

Post-hoc GT evaluation. The output must be a new file, and `--batch` is repeatable:

```bash
~/miniconda3/envs/rigeuler3dgs/bin/python -B tools/evaluate_temporal_local_pilot.py --batch "$OUT" --data-root "$DATA" --output "$OUT/posthoc_gt.json"
```

Gate 3 is prepared but **not run**. Launch it only after deciding on section 10, items 1 and 2:

```bash
bash tools/run_temporal_rgbd_critical.sh --mode local-correlated,simplified --direction forward --frame-limit 11 \
  --scenes USB_100109,blade_103706,storage_45135,stapler_103111 --pcd-subsample-factor 4 --pcd-subsample-method seeded-uniform \
  --pcd-subsample-seed 0 --temporal-warp-noise ideal --temporal-lag-frames 3 --stop-after free --temporal-runtime-budget-seconds 1800
```

Gate 4 is the same command with `--direction backward --scenes USB_100109,blade_103706`.

Gate 5, the 60-frame all-scene batch, is **not authorized**. The command for later reference is:

```bash
bash tools/run_temporal_rgbd_critical.sh --mode local-correlated --direction both --stop-after free --pcd-subsample-factor 1 \
  --temporal-warp-noise <calibrated policy> --temporal-lag-frames 3
```

## 10. Open blockers and limitations

1. **Common-domain acceptance rule** (inherited and shared by all temporal modes). It fixes the current owner pixel at the predicted projection. True motion of about 1.3 px per frame therefore cannot be accepted from a zero-velocity prior. Changing it is a methodological decision, not a covariance one, and was not done.
2. **Warp noise.** `ideal` is uncalibrated. At GT, moving-point depth residuals are 2σ to 10³σ of the measured sensor noise, which is about 0 on Blade. A justified `R_warp`, or a physics review of the depth warp for articulation, is needed before any accuracy claim.
3. **Cross-point covariance is neglected by design.** On the toy shared-pixel case, the within-point marginal differs from the global one by 1e-8 relative. On real scenes it is not measured, because the global exact model is infeasible beyond 3 frames.
4. **Not run:** the legacy comparison, the Storage 8444/45588 equilibrium pair, the 11-frame pilots, the 1k/10k and factors 2/4 performance sweep (only factors 1 and 8 were measured), and an extended-precision arbiter for Blade's ill-conditioned local innovations.
5. **IEKF effort.** About 290 z-buffer and row linearizations per frame. Points rejected at every step keep the 25-step line search running, as in simplified.

**Engineering defects found and fixed during the pilots:**

- Line-search trials registered the raw keys of rejected cells, which gave unbounded working blocks: one block reached dimension 2,388 and caused `LOCAL_RESOURCE_LIMIT`.
- Unbounded batched `eigvalsh` drove the GPU to 7.3 GB. It is now capped at 256 matrices per batch.
- The chunk budget is now truthfully accounted: 24 dense tensors for the posterior pass and 6 for the proposal pass.
- The shared debug exporter needed the per-camera keys.

**HTML diagnostics:** `<batch>/<scene>_local-correlated/free_forward/temporal/index.html` is the live viewer, now including the per-point covariance-retention panel. The full viewer is at `<batch>/<scene>_local-correlated/debug/index.html`.
