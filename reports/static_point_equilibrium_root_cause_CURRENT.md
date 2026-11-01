# Static-point equilibrium failure — current-code audit (2026-10-29, HEAD e1e663b)

This report answers `RGBDiekf_Codex_Static_Point_Equilibrium_Root_Cause_Audit.md`. It is diagnostic only: inference code, policy, noise and defaults are unchanged. Every alternative below is either an offline evaluation of the measurement function or a single linearized step, and none of them is published as a state.

**Code fingerprint:**
- HEAD `e1e663b`; no tracked file is modified.
- New untracked files: `tools/audit_static_equilibrium.py` (sha256 `6982661b…`), `tests/test_static_equilibrium_audit.py`, and this report with its trace.
- `estimator/point_kalman_cuda.py`: sha256 `2f7382ca…`.
- `estimator/projective_terms.py`: sha256 `8e6bb4e6…`.

## 1. Executive verdict

Demonstrated with the current code, on 3 synthetic scenes, the static fixture and 2 real scenes:

1. **E1, the measurement pushes a correctly predicted static point.** At frame 1 the prediction equals the truth exactly. Even so, the first Gauss–Newton step from the truth has a median of 0.4–7 mm and a p90 of 1.4–18 mm. The causes are a systematic depth bias on the synthetic renders and cross-view colour mismatch. Even unbiased noise moves the point, because the prior covers 1.7–4 cm.
2. **E4, all of that correction becomes velocity.** Δv = P_vx P_xx⁻¹ Δx holds to ≤ 7e-8 relative. dt·L = 1.000 at frame 1, because 99% of P⁻_xx is dt²·P_vv, and P_vv is the UNCALIBRATED 1 (m/s)².
3. **E2, steps leave the bilinear model.** Onset iterations cross pixel cells (14–20 of ~20 per scene). They do not converge, and the cell at a pixel corner is chosen by round-off.
4. **E5, persistence.** Velocity injections explain a median 94–98% of the final drift of departing static points; repeated position corrections explain 2–6%. The next frames are mostly prediction-only, or their accepted updates increase the error.
5. **E3, secondary.** In 2–8 of ~20 onsets per scene, the accepted state is worse than the prior on the same rows. Consistency acceptance is not monotone.
6. **E6, NOT REPRODUCED.** No accepted row of a hidden point was found (850 rows checked).

**Unknown:**
- Real scenes have no ground truth. A depth-constant proxy over the first 2 s is used, and it cannot exclude a surface sliding along itself.
- Why each synthetic depth row is biased is only partly established. The sign is uniformly negative, which is consistent with bilinear interpolation of Z being convex on slanted planes (not proven), and with 2×2 cells spanning edges or creases (shown).

## 2. Reproduction manifest

Outputs are in `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/static_equilibrium_audit_20261029_1718/<scene>/{baseline,traced}`. Logs sit next to them; there is no code in outputs.

**Environment:** Python 3.11.15, torch 2.11.0+cu128, CUDA 12.8, RTX 4070 Laptop (8 GB, shared).

**The six required functions exist under their names:**
- `scene/endpoint.py::endpoint_samples`
- `estimator/free_projective_run.py::run_free`
- `estimator/point_kalman_cuda.py::{predict, PointMeasurement, update_points}`
- `estimator/projective_terms.py::build_terms`
- `estimator/projective_sensor.py::sample`
- `estimator/projective_visibility.py::{visibility, apply_frozen_assignment}`

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf; PY=~/miniconda3/envs/rigeuler3dgs/bin/python
OUT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/static_equilibrium_audit_20261029_1718"
# 1. untraced baseline (fixture: add --fps 30; scene from tools/audit_static_equilibrium.py fixture "$OUT/static_fixture")
$PY train.py --scene "$SC" --output "$OUT/$N/baseline" --direction forward --stop-after free --debug none --no-evaluate --no-qa --debug-frames ""
# 2. offline selection (GT / proxy used here only), population tables
$PY tools/audit_static_equilibrium.py select "$OUT/$N/baseline" --scene "$SC" --kind synthetic   # fixture | real --window 50
$PY tools/audit_static_equilibrium.py onset "$OUT/$N/baseline"; $PY tools/audit_static_equilibrium.py population "$OUT/$N/baseline"
# 3. same run, traced on the selected ids / frames (printed by select), then the analysis
$PY train.py ...same... --output "$OUT/$N/traced" --debug-point-ids <ids> --debug-frames <frames>
$PY tools/audit_static_equilibrium.py analyze "$OUT/$N/traced" --baseline "$OUT/$N/baseline"
```

**Contract.** These are the defaults; nothing is overridden.
- `--sensor-noise measured`: R measured per scene.
- `update_acceptance = last_consistent_iterate`, `row_innovation_gate = rgbd_chi2`, `final_visibility = recheck_self_occlusion`.
- `max_iekf_iters` 10, `pixel-reuse-assignment per-frame`, float64, `rendered_mask_measurement off`.
- Every point is kept (subsample factor 1).

**Prior:**
- P_xx0: the back-projection covariance of the source pixel; pixel variance 1/12 px², and depth variance from R (2.6–9.3e-6 m² trace).
- P_xv0 = 0.
- P_vv0 = 1 (m/s)² I and q_c = 0.01 m²/s³ I: both `documented-initial-guess`, **UNCALIBRATED** (v99 not declared).

**Frame times from metadata:**

| scenes | dt | frames | source |
|---|---|---|---|
| synthetic | 1/59 s | 60 | `metadata/timing.json` |
| real | 0.04 s | 390 / 391 | `metadata/frame_timestamps.csv` |
| fixture | 1/30 s | 6 | `--fps 30`, as its tests declare |

**Measured R:**

| scene | σ_rgb | σ_D |
|---|---|---|
| storage | 0.0094 | **5.7 µm** (noiseless render: quantization floor plus cross-camera model variance) |
| USB | 0.0042 | 55 µm |
| stapler | 0.0051 | 12 µm |
| fixture | 0.0081 | 17 µm |
| laptop | 0.017 | 3.4 cm |
| black_box | 0.028 | 2.0 cm |

**Static basis (offline only):**
- **Synthetic:** GT part labels from `points3D_parts.ply`, 5,344 points, spacing h = 1.8 cm.
  - GROUNDED STATIC means the nearest labelled point is static and within h, and no moving-labelled point lies within 2h. The rest are UNGROUNDED and excluded (storage: 202,439 static, 21,689 moving, 122,309 ungrounded).
  - x_GT = x0 at every frame (identity transform). For these points the GT error therefore equals the drift from the canonical point. How well x0 itself is placed is checked at frame 0 (§4).
- **Fixture:** static by construction.
- **Real (README: "Masks and joint GT are absent"):** a labelled **DEPTH-CONSTANT PROXY**.
  - Definition: the nearest pixel of x0 stays valid foreground in ≥ 2 cameras, with |D_k − D_0| ≤ 3√2 σ_D, in every frame 0–50 (2.0 s). The audit stops at frame 50.
  - Colour is not required: with the measured RGB σ, no laptop pixel keeps its colour over the sequence (0% in every camera).
  - Result: 3,882 laptop and 6,215 black_box points.

**Audit samples:** 22 per scene.
- the 10 worst final errors;
- 5 stable controls (never > 1 cm), matched on rows and depth;
- 2 points never updated;
- 5 with the highest frame-1 NIS.

Trace frames are frames 1–5 plus ±2/+3 around each event. Traced and untraced runs are **bit-identical in all 17 arrays, in all six scenes** (spec test 7).

## 3. Baseline and population results (all static points, not only samples)

### Baseline (untraced, forward, current defaults, mask off)

| scene | static points (basis) | final error p50 / p90 / p99 | > 10 cm | final speed p50 / max (truth 0) | point-frames without rows | runtime |
|---|---|---|---|---|---|---|
| fixture | 54814 ({'grounded_static': 54814}) | 0.3 / 2 / 7 mm | 0.0% | 0.000 / 0.28 m/s | 34% | 3 s |
| storage | 202439 ({'grounded_static': 202439, 'grounded_moving': 21689, 'ungrounded': 122309}) | 13.9 / 686 / 1940 mm | 31.4% | 0.009 / 5.85 m/s | 49% | 104 s |
| USB | 34796 ({'grounded_static': 34796, 'grounded_moving': 22037, 'ungrounded': 39134}) | 13.6 / 224 / 1167 mm | 17.8% | 0.011 / 3.22 m/s | 55% | 37 s |
| stapler | 50024 ({'grounded_static': 50024, 'grounded_moving': 35531, 'ungrounded': 51631}) | 75.9 / 633 / 1871 mm | 45.0% | 0.067 / 5.23 m/s | 58% | 42 s |
| laptop (proxy) | 3882 ({'proxy_static': 3882, 'one_camera_constant': 15145, 'none': 13479, 'also_colour_constant_in_2_cameras_(not_required)': 684}) | 230.7 / 903 / 2236 mm | 66.6% | 0.127 / 2.55 m/s | 63% | 161 s |
| black_box (proxy) | 6215 ({'proxy_static': 6215, 'one_camera_constant': 16870, 'none': 10824, 'also_colour_constant_in_2_cameras_(not_required)': 0}) | 172.0 / 909 / 2394 mm | 62.3% | 0.102 / 3.38 m/s | 70% | 170 s |

### Population: the update at each static point's FIRST velocity injection

| scene | group | points | first step p50 | rows 1 / 2 / ≥3 | converged | 10 iterations | \|δx\| p50 / p90 | \|δv\| p50 / p90 | \|δv\|·dt/\|δx\| p50 |
|---|---|---|---|---|---|---|---|---|---|
| fixture | departs > 10 cm | 0 | — | — | — | — | — | — | — |
| fixture | stays < 1 cm | 43865 | 1 | 30614 / 11392 / 1859 | 92% | 9% | 0.37 / 1.3 mm | 0.010 / 0.04 m/s | 0.999 |
| storage | departs > 10 cm | 63520 | 1 | 36800 / 22937 / 3783 | 33% | 69% | 8.18 / 29.1 mm | 0.411 / 1.24 m/s | 0.989 |
| storage | stays < 1 cm | 73905 | 1 | 53869 / 17671 / 2365 | 74% | 27% | 0.53 / 2.9 mm | 0.020 / 0.14 m/s | 0.989 |
| USB | departs > 10 cm | 6202 | 1 | 2995 / 2680 / 527 | 68% | 33% | 2.88 / 17.0 mm | 0.130 / 0.69 m/s | 0.990 |
| USB | stays < 1 cm | 9944 | 1 | 8214 / 1680 / 50 | 93% | 7% | 0.00 / 0.1 mm | 0.000 / 0.00 m/s | 0.992 |
| stapler | departs > 10 cm | 22522 | 1 | 11473 / 9236 / 1813 | 61% | 40% | 3.41 / 24.8 mm | 0.148 / 0.84 m/s | 0.988 |
| stapler | stays < 1 cm | 7847 | 1 | 6795 / 988 / 64 | 93% | 7% | 0.00 / 1.2 mm | 0.000 / 0.04 m/s | 0.988 |
| laptop (proxy) | departs > 10 cm | 2586 | 1 | 1051 / 1535 / 0 | 21% | 80% | 9.70 / 24.2 mm | 0.185 / 0.44 m/s | 0.999 |
| laptop (proxy) | stays < 1 cm | 19 | 3 | 16 / 3 / 0 | 58% | 42% | 0.93 / 2.4 mm | 0.005 / 0.02 m/s | 0.200 |
| black_box (proxy) | departs > 10 cm | 3873 | 1 | 2265 / 1563 / 45 | 28% | 73% | 6.15 / 29.8 mm | 0.115 / 0.53 m/s | 0.997 |
| black_box (proxy) | stays < 1 cm | 83 | 2 | 74 / 9 / 0 | 60% | 40% | 0.84 / 2.6 mm | 0.005 / 0.04 m/s | 0.500 |

### Population: the frame AFTER the first injection

| scene | group | no valid rows | rejected | accepted, error down | accepted, error up | prediction displacement p50 / p90 | most frequent non-VALID reasons |
|---|---|---|---|---|---|---|---|
| fixture | stable | 20% | 0% | 73% | 7% | 0.3 / 1.2 mm | DUPLICATE_PIXEL_SUPPRESSED 101631, SELF_OCCLUDED 22070, DEPTH_DISCONTINUITY 428 |
| storage | departs | 35% | 4% | 25% | 36% | 7.0 / 20.9 mm | SELF_OCCLUDED 294894, DUPLICATE_PIXEL_SUPPRESSED 146330, INNOVATION_INCONSISTENT 3999 |
| storage | stable | 32% | 1% | 49% | 19% | 0.3 / 2.3 mm | SELF_OCCLUDED 341078, DUPLICATE_PIXEL_SUPPRESSED 177006, MEASURED_SURFACE_IN_FRONT 1672 |
| USB | departs | 31% | 2% | 24% | 43% | 2.2 / 11.7 mm | SELF_OCCLUDED 24479, DUPLICATE_PIXEL_SUPPRESSED 17479, INNOVATION_INCONSISTENT 714 |
| USB | stable | 14% | 0% | 52% | 34% | 0.0 / 0.0 mm | SELF_OCCLUDED 38908, DUPLICATE_PIXEL_SUPPRESSED 29057, INNOVATION_INCONSISTENT 984 |
| stapler | departs | 30% | 6% | 25% | 39% | 2.5 / 14.2 mm | DUPLICATE_PIXEL_SUPPRESSED 79307, SELF_OCCLUDED 72798, INNOVATION_INCONSISTENT 2223 |
| stapler | stable | 30% | 1% | 47% | 22% | 0.0 / 0.6 mm | DUPLICATE_PIXEL_SUPPRESSED 28273, SELF_OCCLUDED 26962, INNOVATION_INCONSISTENT 575 |
| laptop (proxy) | departs | 32% | 1% | 21% | 47% | 7.4 / 17.7 mm | DUPLICATE_PIXEL_SUPPRESSED 3735, MASK_EXCLUDED 822, INVALID_DEPTH 318 |
| laptop (proxy) | stable | 37% | 0% | 16% | 47% | 0.2 / 0.8 mm | DUPLICATE_PIXEL_SUPPRESSED 30, INVALID_DEPTH 6, MASK_EXCLUDED 5 |
| black_box (proxy) | departs | 41% | 1% | 17% | 40% | 4.6 / 21.1 mm | DUPLICATE_PIXEL_SUPPRESSED 6032, MASK_EXCLUDED 1280, INVALID_DEPTH 760 |
| black_box (proxy) | stable | 38% | 5% | 17% | 40% | 0.2 / 1.5 mm | DUPLICATE_PIXEL_SUPPRESSED 144, MASK_EXCLUDED 18, INNOVATION_INCONSISTENT 12 |

### Population: trajectory attribution of points that depart > 10 cm (exact identity, projected on the net drift)

| scene | points | velocity injections share p10 / p50 / p90 | position corrections share p50 | first injection alone p50 | identity closure (max) |
|---|---|---|---|---|---|
| fixture | 0 | — | — | — | — |
| storage | 63520 | 0.92 / 0.98 / 1.06 | 0.02 | 0.87 | 5.3e-15 m |
| USB | 6202 | 0.71 / 0.94 / 1.02 | 0.06 | 0.24 | 3.6e-15 m |
| stapler | 22522 | 0.71 / 0.95 / 1.03 | 0.05 | 0.21 | 5.3e-15 m |
| laptop (proxy) | 2586 | 0.86 / 0.96 / 1.01 | 0.04 | 0.56 | 5.8e-15 m |
| black_box (proxy) | 3873 | 0.78 / 0.95 / 0.98 | 0.05 | 0.43 | 4.9e-15 m |

### Population: the measurement AT THE STATIC LOCATION (v = 0, no update; production measurement code)

| scene | frame | points with rows | score/dof p50 | rows of points above χ²₉₉ (1% expected) | first step from the truth p50 / p90 | implied velocity p50 / p90 | mean r/σ per camera, depth |
|---|---|---|---|---|---|---|---|
| fixture | 1 | 37653 | 0.43 | 3.9% | 0.37 / 1.4 mm | 0.011 / 0.04 m/s | -0.38 … -0.36 |
| fixture | 5 | 37653 | 0.43 | 3.9% | 0.37 / 1.4 mm | 0.011 / 0.04 m/s | -0.38 … -0.36 |
| storage | 1 | 141977 | 0.62 | 17.0% | 2.84 / 11.4 mm | 0.166 / 0.66 m/s | -31.62 … -0.54 |
| storage | 59 | 139688 | 0.61 | 16.9% | 2.84 / 11.4 mm | 0.166 / 0.67 m/s | -16.94 … 13.08 |
| USB | 1 | 22386 | 0.01 | 13.8% | 0.01 / 4.2 mm | 0.000 / 0.24 m/s | -5.79 … -0.93 |
| USB | 59 | 21033 | 0.01 | 15.3% | 0.01 / 4.7 mm | 0.000 / 0.27 m/s | -5.72 … -0.41 |
| stapler | 1 | 32623 | 0.61 | 39.7% | 1.21 / 12.0 mm | 0.071 / 0.70 m/s | -16.27 … -10.67 |
| stapler | 59 | 32230 | 0.66 | 40.2% | 1.26 / 12.2 mm | 0.074 / 0.71 m/s | -16.53 … -8.08 |
| laptop (proxy) | 1 | 2396 | 0.17 | 6.3% | 6.91 / 14.6 mm | 0.172 / 0.37 m/s | -2.67 … 0.01 |
| laptop (proxy) | 5 | 2391 | 0.18 | 5.0% | 6.97 / 15.1 mm | 0.174 / 0.38 m/s | -1.63 … -0.01 |
| black_box (proxy) | 1 | 3702 | 0.10 | 5.3% | 3.42 / 17.8 mm | 0.085 / 0.44 m/s | -0.24 … -0.10 |
| black_box (proxy) | 5 | 3703 | 0.12 | 5.0% | 3.76 / 18.8 mm | 0.094 / 0.47 m/s | -0.23 … -0.09 |

**How to read these tables:**
- **Fixture:** no point departs, but the frame-1 first step from the truth still has a median of 0.37 mm.
- **Departing vs stable:** the row count does not separate them, since both are mostly one- or two-row updates. Departing points have a larger first correction (median 3–10 mm against 0–1 mm) and are mostly non-converged (21–68% converged against 58–93%).
- **The frame after the injection** does not take the velocity back. Departing points get either no rows (30–41%) or accepted updates that increase the error more often than they decrease it.
- **After a > 10 cm departure** most point-frames are `NO_VALID_ROWS`. Storage: 1.68 M of 2.72 M; the others are similar (`after_departure_acceptance_mix` in `selection.json`).

## 4. Onset table

### Onset table (traced audit points: the 5 worst per scene; all 132 audit points in static_onset_trace.jsonl)

| scene | point (src cam, px) | 1st correction / error increase / velocity / > 10 cm (frame) | rows at s⁻ → accepted | dominant row (one-step split) | E1: χ² at truth (dof) | 1st step from truth | iterate accepted / run, converged | max px step | \|δx\| published | \|δv\| | max dt·L σ | accepted vs prior, common rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture | 49408 (c3, 64,133) | 1 / 1 / 1 / None | [1] → [1] | cam1_G | 0.934 (3) | 1.3 mm | 10 / 10, no | 0.3 | 9.4 mm | 0.28 m/s | 1.000 | 6.63 → 3.68 |
| fixture | 22001 (c1, 64,133) | 1 / 1 / 1 / None | [3] → [3] | cam3_G | 0.934 (3) | 1.3 mm | 10 / 10, no | 0.3 | 9.4 mm | 0.28 m/s | 1.000 | 6.63 → 3.68 |
| fixture | 6657 (c0, 141,131) | 1 / 1 / 1 / None | [1, 2] → [1] | cam2_B | 12.8 (3) | 2.2 mm | 9 / 9, yes | 0.6 | 9.0 mm | 0.27 m/s | 1.000 | 5.35 → 0.8 |
| fixture | 1378 (c0, 221,90) | 1 / 1 / 1 / None | [3] → [3] | cam3_B | 27.4 (3) | 11.4 mm | 8 / 8, yes | 1.3 | 8.0 mm | 0.24 m/s | 1.000 | 33.7 → 13.6 |
| fixture | 28785 (c2, 221,90) | 1 / 1 / 1 / None | [1] → [1] | cam1_B | 27.4 (3) | 11.4 mm | 8 / 8, yes | 1.3 | 8.0 mm | 0.24 m/s | 1.000 | 33.7 → 13.6 |
| storage | 209203 (c4, 319,408) | 1 / 1 / 1 / 2 | [3] → [3] | cam3_R | 23.6 (3) | 25.4 mm | 7 / 10, no | 4.0 | 48.7 mm | 2.85 m/s | 1.000 | 23.6 → 11.8 |
| storage | 16209 (c0, 316,292) | 2 / 2 / 2 / 2 | [0, 7] → [0] | cam7_R | 21.9 (3) | 5.3 mm | 10 / 10, no | 20.8 | 120.5 mm | 3.55 m/s | 0.500 | 8.5e-12 → 14 |
| storage | 183223 (c4, 343,220) | 1 / 1 / 1 / 2 | [3, 4, 5] → [3, 4, 5] | cam5_B | 36.6 (3) | 37.5 mm | 7 / 10, no | 6.3 | 67.4 mm | 3.94 m/s | 1.000 | 52.5 → 19.5 |
| storage | 159326 (c3, 238,327) | 1 / 1 / 1 / 2 | [1, 2] → [1, 2] | cam1_B | 16.5 (3) | 8.0 mm | 10 / 10, no | 7.3 | 58.4 mm | 3.40 m/s | 1.000 | 16.7 → 16.8 |
| storage | 203530 (c4, 280,365) | 1 / 1 / 1 / 2 | [3, 4] → [3, 4] | cam3_B | 6.7 (3) | 5.0 mm | 10 / 10, no | 5.5 | 64.9 mm | 3.79 m/s | 1.000 | 36.5 → 16.8 |
| USB | 34397 (c2, 385,327) | 21 / 21 / 21 / 21 | [6] → [6] | cam6_D | 4.16e+08 (2) | 1144.8 mm | 3 / 3, yes | 12.4 | 1144.8 mm | 3.22 m/s | 0.048 | 4.16e+08 → 12 |
| USB | 46189 (c3, 243,325) | 21 / 21 / 21 / 21 | [6] → [6] | cam6_D | 4.18e+08 (2) | 1144.6 mm | 4 / 4, yes | 11.2 | 1144.6 mm | 3.22 m/s | 0.048 | 4.18e+08 → 10.3 |
| USB | 49526 (c4, 222,270) | 1 / 1 / 1 / 3 | [1, 2] → [1, 2] | cam2_G | 1.38e+04 (3) | 15.2 mm | 9 / 10, no | 15.8 | 44.9 mm | 2.64 m/s | 1.000 | 1.38e+04 → 15.9 |
| USB | 52645 (c4, 390,281) | 1 / 1 / 1 / 2 | [3, 4, 5] → [3, 4, 5] | cam3_D | 1.06e+03 (3) | 9.0 mm | 10 / 10, no | 4.6 | 53.9 mm | 3.14 m/s | 1.000 | 1.09e+03 → 10.5 |
| USB | 60847 (c4, 337,311) | 1 / 1 / 1 / 2 | [3, 4] → [3, 4] | cam3_D | 287 (3) | 6.8 mm | 5 / 10, no | 7.9 | 53.4 mm | 3.12 m/s | 1.000 | 287 → 11.3 |
| stapler | 111800 (c6, 304,284) | 1 / 1 / 1 / 2 | [4, 6] → [4] | cam4_D | 7.9e+03 (3) | 25.3 mm | 10 / 10, no | 5.5 | 65.3 mm | 3.79 m/s | 1.000 | 7.93e+03 → 15.8 |
| stapler | 133372 (c7, 297,314) | 2 / 2 / 2 / 3 | [1] → [1] | cam1_G | 53 (3) | 33.0 mm | 10 / 10, no | 2.7 | 47.7 mm | 1.41 m/s | 0.500 | 53 → 8.75 |
| stapler | 84084 (c4, 194,301) | 3 / 3 / 3 / 3 | [2, 3, 4] → [3, 4] | cam2_D | 7.7e+03 (3) | 248.7 mm | 9 / 10, no | 42.4 | 226.6 mm | 4.45 m/s | 0.333 | 1.48 → 22.5 |
| stapler | 133608 (c7, 356,315) | 1 / 1 / 1 / 2 | [0, 1, 7] → [0, 1, 7] | cam1_D | 3.2e+05 (3) | 5.4 mm | 7 / 10, no | 12.3 | 66.9 mm | 3.93 m/s | 1.000 | 3.2e+05 → 19.4 |
| stapler | 80496 (c4, 192,280) | 1 / 1 / 1 / 2 | [3] → [3] | cam3_R | 2.11 (2) | 0.2 mm | 2 / 2, yes | 0.0 | 0.2 mm | 0.01 m/s | 1.000 | 6.89 → 4.78 |
| laptop (proxy) | 26068 (c2, 291,302) | 1 / 1 / 1 / 2 | [1, 2] → [1, 2] | cam1_D | 0.959 (3) | 8.2 mm | 10 / 10, no | 5.9 | 8.2 mm | 0.21 m/s | 1.000 | 1.75 → 1.18 |
| laptop (proxy) | 27247 (c2, 299,311) | 1 / 1 / 1 / 2 | [1, 2] → [1] | cam1_D | 0.403 (2) | 12.2 mm | 10 / 10, no | 21.9 | 88.1 mm | 2.20 m/s | 1.000 | 0.456 → 10.8 |
| laptop (proxy) | 22409 (c2, 299,274) | 1 / 1 / 1 / 2 | [1, 2] → [1] | cam1_R | 19.3 (3) | 29.3 mm | 9 / 10, no | 28.5 | 79.6 mm | 1.99 m/s | 1.000 | 23.9 → 4.73 |
| laptop (proxy) | 25942 (c2, 295,301) | 1 / 1 / 1 / 2 | [1, 2] → [1, 2] | cam1_D | 0.382 (2) | 11.7 mm | 10 / 10, no | 6.5 | 10.7 mm | 0.27 m/s | 1.000 | 1.59 → 3.89 |
| laptop (proxy) | 25685 (c2, 299,299) | 1 / 1 / 1 / 2 | [2] → [2] | cam2_D | 0.0035 (1) | 0.9 mm | 10 / 10, no | 0.4 | 0.9 mm | 0.02 m/s | 1.000 | 0.834 → 0.858 |
| black_box (proxy) | 6222 (c0, 433,389) | 1 / 1 / 1 / 1 | [0] → [0] | cam0_B | 0.23 (2) | 2.7 mm | 7 / 10, no | 88.8 | 135.2 mm | 3.38 m/s | 1.000 | 0.23 → 13.1 |
| black_box (proxy) | 8201 (c0, 412,404) | 2 / 2 / 1 / 2 | [0, 2] → [2] | cam0_D | 56.4 (3) | 124.6 mm | 6 / 10, no | 51.0 | 200.4 mm | 2.50 m/s | 0.503 | 62.8 → 6.49 |
| black_box (proxy) | 4242 (c0, 446,375) | 1 / 1 / 1 / 2 | [2] → [0] | cam2_B | 52.6 (2) | 97.7 mm | 9 / 10, no | 32.7 | 97.7 mm | 2.44 m/s | 1.000 | 0 → 5.96 |
| black_box (proxy) | 5484 (c0, 442,384) | 1 / 1 / 1 / 2 | [0, 1, 2] → [1] | cam2_B | 34.9 (3) | 102.6 mm | 3 / 10, no | 42.4 | 94.0 mm | 2.34 m/s | 1.000 | 15.4 → 16.1 |
| black_box (proxy) | 4110 (c0, 444,374) | 2 / 2 / 2 / 2 | [0, 1, 2] → [0] | cam2_G | 42.3 (3) | 36.2 mm | 7 / 10, no | 40.2 | 177.3 mm | 2.22 m/s | 0.500 | 0.00246 → 9.23 |

**Column notes:**
- "1st step from truth" is (Λ⁻ + I)⁻¹[g, 0] evaluated at x_GT with this frame's P⁻. At frame 1 that is iteration 0 of the production update exactly, because x⁻ = x_GT.
- "Dominant row" comes from the exact additive split of that one step.

**Frame 0, same instant (`frame0_same_instant_check`):**
- rgb0 is the measured source-pixel colour for all 132 audit points.
- At x0 the source camera's own residual is 0 to round-off.
- Other cameras disagree by up to 10⁵ σ in depth. Those are other surfaces at that pixel: occluded views, which the visibility rules exclude at frame 1.

## 5. Worked examples (one per scene; numbers from `analysis.json`)

### Storage, point 59749 (source camera 1, pixel 222,301), frame 1 (t = 1/59 s)

1. **Prior.** s⁻ = x0, v⁻ = 0. P⁻_xx = 2.9e-4 m² I, of which 8.62e-4 m² of trace (99.1%) is dt²·P_vv and 7.4e-6 m² is the source covariance. P⁻_vx = 0.0508 I, of which dt·P_vv = 0.0508 and Q_vx = 4e-6. L = P_vx P_xx⁻¹ ≈ 58.6 s⁻¹ I, and dt·L has singular values (1.000, 0.987, 0.987).
2. **Rows at s⁻ = x_GT.**

   | camera | position | residual | H_D | g_D | NIS (prior-linearized) |
   |---|---|---|---|---|---|
   | 0 | u, v = 264.911, 314.885; du, dv = 0.911, 0.885 | 2×2 depths 3.4623 / 3.4733 / 3.4629 / 3.4733 m: spread 1.1 cm, allowance 4.8 cm, so "same surface". Z = 3.46766, D = 3.47237, **r_D = −4.7 mm = −822 σ** (σ_D = 5.7 µm); RGB r/σ = (0.9, 2.2, 1.9) | (0.83, 1.75, 0.074) | (−1.2e8, −2.5e8, −1.1e7) | 0.14, because √S_zz = 3.3 cm comes from the prior: **gate passed** |
   | 1 (source) | the pixel centre, du = dv = 0 | 0 | — | — | — |

3. **Iterations.** J = 676,314 → 268,982 → 4,984 → 415 → 18.9 → 18.9 (converged).
   - The first step is 63.5 mm. The linearization error at α = 1 is −377 / +361 σ (whitened depth), with a cell change in both cameras; ρ = 0.6.
   - The accepted final iterate is **70.8 mm** away with J = 18.9 ≤ χ²₈ = 23.6. On the common rows, the prior's 676,314 (depth 676,305) becomes 18.9 (prior penalty 17.2).
4. **Velocity.** Δv = (0.089, −0.020, −4.126) m/s = L·Δx, relative error 2.9e-15.
5. **Next prediction.** Error 70.8 → 140.8 mm, against 70.8 mm had no velocity been injected. Frames 2–5 are `NO_VALID_ROWS` (DUPLICATE / SELF_OCCLUDED / INNOVATION_INCONSISTENT) and the point travels ballistically: 210.7, 280.7, 350.6 mm. Final drift 4.127 m, of which 4.057 m comes from the first injection (identity closure 4.4e-15 m).

### Fixture, point 49408 (camera 3, pixel 64,133), frame 1

- **Row:** only camera 1, RGB r/σ = (−0.31, 1.34, −2.13); χ²₃ at the truth = 0.93 (p = 0.82: noise-like). The first step is 1.28 mm, in-cell: Gauss–Newton predicts the pixel change exactly (−0.058, 0.126 px), and ρ = 1.20, 1.05, 1.01, 0.99 for α = 1 … 1/8.
- **Iterations:** they keep descending, 1.3 → 2.1 → 2.9 → 4.0 → 6.2 → 8.6 mm (J 6.6 → 0.44), then enter a **2-cycle** (8.45 ↔ 9.43 mm, ρ = −9.8 / 1.0).
- **Acceptance:** iterate 10, the worse phase (J 3.68 against 0.44), is accepted (≤ 16.3).
- **Velocity:** Δv = 0.283 m/s = Δx/dt (dt·L = 1.000).
- **Afterwards:** frames 2–4 have no rows (DUPLICATE / SELF_OCCLUDED), and the error grows 9.4 → 18.9 → 28.3 → 37.7 mm.

### USB, point 34397, frame 21 (onset after 20 prediction-only frames)

- **Prior:** with no rows from frame 1 to 20, P⁻_xx grew to 0.127 m² I (σ = 36 cm). The camera-6 row was `MEASURED_SURFACE_IN_FRONT` at frames 19–20.
- **The frame-21 row:** camera 6 samples a surface **1.14 m** in front: r_D = 1.1438 m = 2.0e4 σ. Its NIS is 11.97, and the front test needs ν_D > 3√S_zz = 1.16 m (√S_zz = 0.39 m), so the 1.14 m row is admitted.
- **Update:** one step of 1,144.8 mm is accepted (J 12 ≤ 16.3), with Δv = 3.22 m/s (dt·L = 0.048, because P_vx is diluted by the gap).
- **Interpretation:** a wrong-surface row admitted by a prior inflated over a prediction-only interval (E5 → E1).

### Stapler, point 133608 (source camera 7), frame 1

- **Rows (3 cameras):** cameras 0 and 1 have r_D = −5.4 / −3.9 mm = −457 / −332 σ. RGB is −4.9 / −7.4 σ (the canonical colour differs from these views). Camera 7 is the source, r = 0.
- **Iterations:** the first step from the truth is 5.4 mm, then they oscillate (J 3.2e5 … 2.7 … 9.6e4 …). Iterate 7 (66.9 mm, J 19.4 ≤ 30.1) is accepted, with Δv = 3.93 m/s.
- **Afterwards:** frames 3–5 are `REJECTED_INCONSISTENT`, and the point travels ballistically.

### Laptop (proxy), point 22409, frame 1 (dt = 0.04 s)

- **Prior:** P⁻_xx = 1.6e-3 m² I (σ = 4 cm), of which 99.9% is dt²·P_vv.
- **Rows:** camera 1 RGB r/σ = (−3.2, −2.5, −2.7). Camera 2 is the source, sampled at the pixel corner (du = 5.7e-14).
- **Iterations:** the first step is 29.3 mm, then they wander 12–46 mm with 28 px steps. Iterate 9 (79.6 mm) is accepted, with Δv = 1.99 m/s.
- **Afterwards:** frames 2–5 have no rows (MASK_EXCLUDED / DUPLICATE).

### Black_box (proxy), point 6222, frame 1

- **Row:** only its own source camera 0, at the corner, with r_rgb = 2/255 between frames 0 and 1; χ² = 0.23 (insignificant). The first step is 2.7 mm.
- **Iterations:** iterate 1 has no row, so the iteration returns exactly to s⁻, giving the exact cycle 0 ↔ 2.71 mm. At iterate 5, round-off selects another 2×2 cell at the corner. The point then jumps up to 88 px, and iterate 7 (135.2 mm, J 13.1, against 0.23 at the prior on the same row) is accepted.
- **Velocity:** Δv = 3.38 m/s.
- **Afterwards:** frames 2–4 are MASK_EXCLUDED in all 3 cameras, and the point travels ballistically to 6.75 m.

## 6. Matched stable controls

Controls get the same frame-1 treatment: dt·L = 1, mostly one row, frequently non-converged. The numerical difference is the size of the first correction.

**Storage controls (225566, 178834, 65225, 244853):**
- The χ² at the truth is insignificant (0.3–2.0 for 3 dof).
- The first step from the truth is 1.2–3.0 mm, and the published |δx| is 0.28–3.8 mm (Δv 0.016–0.22 m/s).
- Later accepted updates take the velocity back: their projected velocity contribution is only 0.005–0.015 m.

**Stapler and USB controls (129883, 39262, 19515, 71317, 20405):**
- They are seen only by their own source camera at the pixel centre: r = 0 exactly, so δx = 0.00 mm.
- The worst points instead have a second camera whose sample disagrees at the truth (depth −300 … −800 σ, or colour 2–7 σ), or they sit on the corner knife edge.

**Population:**
- Stable points' first corrections: median 0.0–0.9 mm, p90 0.1–2.9 mm.
- Departing points' first corrections: median 2.9–9.7 mm.

The two never-updated controls per scene stay exactly at x0, with an error of 0.

## 7. Temporal visibility timeline

These checks demonstrate the timeline; they do not assume it.

- **Prediction-only frames** verified for every audit point:
  - from the memmaps, over the whole trajectory: v unchanged exactly, and x = x + dt·v within the float32 storage bound, on 5,163 frames;
  - from the traced frames: s⁺ = s⁻ and P⁺ = P⁻ bit for bit, 2,554 of 2,554.
- **Ballistic propagation** is therefore exact (timelines in §5).
- **Hidden-point updates:**
  - Across the 658 accepted traced linearizations (850 rows), no used row had another transported point statistically in front in its pixel.
  - 337 rows shared their pixel with a point within the uncertainty (DUPLICATE). The per-frame frozen assignment allows this by design.
  - No used row had a measured surface in front of the accepted state by its own residual.
- **Reappearance:**
  - "Recovery" here means the first frame with accepted rows after a gap; storage has 193,796.
  - The recovery update reduces the error in 42–62% of cases (synthetic; fixture 88%) and 47–57% (real). Otherwise it increases it.
  - Departed points (> 10 cm) are mostly not corrected because they have **no valid rows**, rather than because rows are rejected. `NO_VALID_ROWS` outnumbers `REJECTED_INCONSISTENT` 12–36× (synthetic) and 42–67× (real); storage: 1,681,296 against 138,976 of 2,721,960 point-frames.

## 8. Mechanism ranking (current code)

| | Mechanism | Verdict | Evidence (counts; trace ids) |
|---|---|---|---|
| E1 | Non-zero residual / information gradient at the true static location | **CONFIRMED** | (a) Bias beyond R: 3.9% (fixture), 17% (storage), 14% (USB), 40% (stapler) of static points' row sets exceed χ²₉₉ at the truth (1% expected). Synthetic depth is negative in every camera (stapler mean r_D/σ −8 … −16), with tails at 2×2 cells spanning edges (storage 59749: 1.1 cm spread admitted by a 4.8 cm allowance). Real: laptop camera 0 RGB −5 σ. (b) Noise-level rows are enough: 25–61% of traced onset points per scene (fixture 12/20, storage 8/20, USB 5/19, stapler 5/20, laptop 9/18, black_box 11/18) have χ² p ≥ 0.01 at the truth (e.g. fixture 49408, laptop 26068 / 27247 / 25942, black_box 6222), yet first steps reach 1–12 mm. |
| E2 | Steps outside the bilinear model (cell crossing, support change), non-convergence | **CONFIRMED** | 14–20 of ~20 onset frames per scene cross cells. 15–18 have a full step with ρ < 0 on the same rows. 9–17 do not converge in 10 iterations. In-cell steps are accurate (ρ ≈ 1; analytic H against finite differences ≤ 6e-9 relative, 1,049 in-cell checks). Knife edge: the source camera at frame 1 sits exactly on a cell corner; from bit-identical states and rows, production and oracle first steps differ by 4.24 m (storage 263686). 13 traced point-frames have diverging full replays; the 5 inspected all diverge at a corner iterate or after a zero-row return to s⁻. |
| E3 | Accepted iterate worse than the prior on comparable rows | **CONFIRMED, secondary** | Onsets: fixture 2/20, storage 4/20, USB 3/19, stapler 3/20, laptop 8/18, black_box 8/18. Accepted worse than the previous iterate: 5–15 per scene. Acceptance tests J ≤ χ²₄ₘ, and with P⁻_xx dominated by dt²·P_vv a 7 cm move costs a prior penalty of ≈ 17 (storage 59749). |
| E4 | Correct Kalman P_vx turns position into velocity | **CONFIRMED** | Δv = L·Δx: ≤ 7e-8 relative on all 658 accepted traced frames. dt·L = 1.000 at frame 1 (P⁻_vx = dt·P_vv, P⁻_xx ≈ dt²·P_vv). Population \|δv\|·dt/\|δx\| median 0.988–0.999. Departing points: velocity injections explain 94–98% (population median) of the drift, the first injection alone 21–87% (population median). |
| E5 | Long prediction-only or rejected intervals let it persist | **CONFIRMED** | Next frame after the injection: 30–41% no rows. Accepted updates increase the error more often than they decrease it (e.g. storage 36% vs 25%). After a > 10 cm departure, `NO_VALID_ROWS` dominates. A gap also inflates P⁻ until a wrong-surface row is admitted (USB 34397 / 46189: 1.14 m). |
| E6 | Real violation of visibility / foreground invariants | **NOT REPRODUCED** | 0 hidden-point rows used out of 850 accepted rows. 0 rows used with the measured surface in front. Foreground sampling requires 4 white mask pixels; no audit row violated it. |

**Is the algebra wrong? No.**
- At every traced iterate, the independent float64 NumPy oracle reproduces A, b and δ from the dumped rows to ≤ 4e-8 relative in 4 of 6 scenes, ≤ 3e-6 on stapler, and ≤ 4e-5 on USB (ill-conditioned A).
- Exception: δ agrees only to 7e-3 on one USB iterate, at cond(A) = 5×10⁸. b agrees to 1.4e-6 there, because it is a difference of terms weighted 1/σ_D² ≈ 3×10¹⁰ (float64 cancellation).
- That is evidence about the code, not about the model.

## 9. Minimal next diagnostic or correction target (not implemented, separate from the findings)

**Most supported single issue:**
- **Where:** the fixed prior that `estimator/free_projective_run.py::run_free` builds from `config.velocity_covariance`, propagated by `estimator/point_kalman_cuda.py::predict`.
- **What:** at frame 1, P⁻_xx = Σ0 + dt²·P_vv + Q_xx is 99.1–99.9% dt²·P_vv, and P⁻_vx = dt·P_vv. So L = 1/dt, and any accepted correction of a static point is published as v = δx/dt.
- **Why it matters:** this holds whether the correction is caused by bias (E1a), by noise (E1b), or by a non-convergent iterate (E2/E3). Persistence (E5) then propagates it.
- **Who decides:** P_vv is the UNCALIBRATED documented guess, so its value is the user's to declare (`--velocity-prior-speed99`). This audit changes nothing.

**Second, independent issue:**
- **Where:** the measured R on the noiseless synthetic renders (σ_D 5.7–55 µm) against the bilinear depth model error at the truth (mm, with a consistent negative sign): `scene/noise_calibration.py::calibrate`, `estimator/projective_sensor.py::sample`.
- **Why it matters:** the row gate cannot reject those rows, because S is dominated by the same broad prior (√S_zz 2–5 cm).

**Next diagnostic, if wanted:**
- Repeat the at-truth population test (`population`) with the depth rows analytically corrected for the convexity of Z on a plane, to confirm or refute the sign hypothesis.
- Repeat the onset statistics (`onset`) under a declared v99, as a one-variable replay.

## 10. Tests, overhead, data volume

**Tests:**
- `tests/test_static_equilibrium_audit.py` (spec section 7): 9 tests.
  1. a zero-residual static point keeps its mean;
  2. no rows → s⁺ = s⁻, P⁺ = P⁻, prediction by dt·v;
  3. an occluder in one camera removes only that row;
  4. Δv = L·Δx (linear update and accepted IEKF states); the attribution identity closes on non-uniform times;
  5. in-cell linearization is exact and a cell crossing is reported;
  6. oracle acceptance codes and the comparable-row objective match production;
  7. tracing on/off parity through `train.py`.
- Together with `test_root_cause_oracle`, `test_point_kalman_cuda` and `test_projective_measurement`: **33 tests, OK**.
- Parity on the six audited runs: all arrays IDENTICAL.

**Overhead:**
- With tracing off, nothing is allocated and the outputs are identical.
- With tracing on, a traced frame costs about 9 s on storage against about 1.7 s. Whole runs: storage 104 → 329 s, USB 37 → 331 s, stapler 42 → 189 s, laptop 161 → 253 s, black_box 170 → 285 s, fixture 3 → 26 s.

**Data volume:**
- Traces: 40–550 MB of JSON per scene, in the output root only.
- Analyses: 8–42 MB per scene.
- In the repository: only this report and `reports/static_point_equilibrium/static_onset_trace.jsonl` (132 lines, 288 KB).

**Not done or limited:**
- Real-scene static identity is a proxy, not GT.
- The GT labels are sparse, so 35–51% of synthetic points are UNGROUNDED and excluded.
- Counterfactuals are single linearized steps (they do not model correlated rows or the nonlinear path).
- The backward direction was not audited (the spec asks for forward only).
