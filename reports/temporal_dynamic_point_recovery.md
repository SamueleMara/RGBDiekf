# Material-following temporal RGB-D IEKF — moving-point recovery, USB_100109 and blade_103706, frames 0–2

**Status: `DYNAMIC_NOT_RECOVERED`** on both scenes (partial on USB), together with `IMAGE_MOTION_UNOBSERVABLE` for most moving points and `COVARIANCE_INCONSISTENT` for moving points at frame 2. `MATERIAL_WARP_INVALID_AT_GT` is **not** triggered.

- Static accuracy is preserved exactly: static drift is 0.000 mm at p99 on both scenes, against 16.6 / 36.4 mm for legacy.
- Machine-readable results are in `reports/temporal_dynamic_point_recovery.json`.
- The GT residual audit is in `reports/temporal_dynamic_gt_warp_audit.md`.
- Nothing is committed. The 60-frame / ten-scene batch was **not** launched: the spec's precondition (frame-2 moving points move toward GT) is not met.

## 1. What was implemented (opt-in `--temporal-mode material-following`; legacy / simplified / full / local-correlated unchanged)

| Equation / rule | Code |
|---|---|
| q_t = p_ref + π(x_t) − π(x_ref) (moving current sample, fixed measured reference pixel) | `estimator/temporal_material_terms.py:185` (`build_material_rows`) |
| f = Σ w_a(y_a−η_a) − (y_ref−η_ref) − e₄ΔZ | `temporal_material_terms.py:103` (`material_prediction`) and `temporal_material_iekf.py:68` (`linearize`) |
| ∂f/∂x_t = G J_π,t − e₄J_Z,t; ∂f/∂x_ref = −G J_π,ref + e₄J_Z,ref; velocity columns zero | `temporal_material_terms.py:110-113` |
| ∂f/∂η_ref = +I₄, ∂f/∂η_t,a = −w_a I₄ in the stacked local system | `temporal_material_iekf.py:133,136` (`_system`) |
| Reference raw error = the same local variable (the source key at frame 0; the retained key afterwards) | `linearize` key lookup; `material_retention_mask` `temporal_material_terms.py:258` |
| Current support validity: mask, depth, single surface | `validate_reference_patch` call, `temporal_material_terms.py:186` |
| First-hit/occlusion at every candidate state (self, measured, at the sample) | `temporal_material_terms.py:176,178,198` |
| Reference commit only on the point's own measured surface, after an accepted update or a frame without rows | `temporal_material_terms.py:236` |
| Expiry ledger per needed key (never blocks never-used pixels) | `temporal_material_terms.py:278` |
| Acceptance objective ½δᵀz + Σ[½e₀ᵀR⁻¹e₀ + ½log det R] (current errors integrated out) | `temporal_material_iekf.py:167,179` |
| Material-identity acceptance (no current-pixel signature) | `temporal_material_iekf.py:194` (`compare`: 200 identity loss, 204 background, 206 common identity, 209 descent) |
| Stationary / orphan handling | `temporal_material_iekf.py:331,376,386` |
| Joseph covariance and exact local marginalization (shared, with three behaviour-preserving hooks) | `temporal_local_iekf.py:455-464,500` |
| Optional coarse-to-fine initializer (B1 pyramid) | `temporal_material_pyramid.py:27,49,119`; `temporal_material_iekf.py:245` |
| Optional data-derived resampling discrepancy (separate from R_sensor and R_warp) | `temporal_material_terms.py:117` |

- The old fixed-current-pixel row signature (`point_domains`, `same_current_domain`, `row_signature`) is used **0 times** in the material modules.
- The local runner gained a `variant` hook (`LocalVariant` default = byte-for-byte the previous local-correlated behaviour; its 31 tests still pass).
- New CLI flags (train.py and the batch tool, forwarded to material children only):
  - `--temporal-material-resampling {none,second-difference}` (default `none`);
  - `--temporal-material-pyramid-levels N` (default 0).

**Fixes made during the work** (each found by a test or diagnostic):
- **Occluder at the sample.** The current sample could lie on a nearer occluder while the point's own pixel did not, so the depth test is now also applied at q_t.
- **Log-determinant in the acceptance objective.** With weight-dependent R, the objective without log det prefers integer pixel positions.
- **Accepted points left with no valid identity.** Such points are reset to their exact prediction ("orphans").
- **Points already at a stationary iterate** are conditioned rather than left prediction-only.

## 2. Tests

The full focused suite (`python -B tools/test_temporal_rgbd.py`) passes **129/129**: 106 previous + 23 new, 0 skips, on CPU and CUDA, with no source changes during the run (`tests/temporal_test_results.json`). The new tests are:

- `test_temporal_material_warp` (3):
  - residual, q and weights equal the independent NumPy oracle to 1e-12;
  - H_t and H_ref match 5-point finite differences to ≤ 9.1e-10, at ±1.3 / ±3 px, with depth motion, subpixel source offsets and two cameras;
  - velocity columns are exactly 0;
  - η Jacobians are +I and −w_a I;
  - at a cell transition the residual jumps by only 5.5e-9 while the Jacobian jumps by 0.11 (the new piece).
- `test_temporal_material_association` (7):
  - ±1.3 px, ±3 px and depth motion are recovered on exactly rendered scenes (max error 0.023 px at frame 2, velocity within 5%), while local-correlated stays 0.80–1.28 px off;
  - static scenes stay static to 1e-12;
  - no background sample or update;
  - deleting or losing a measurement never creates descent;
  - occlusion keeps the reference and the identity reappears at age 2;
  - a same-surface collision keeps its identity;
  - two-camera bootstrap/handover works.
- `test_temporal_material_local_covariance` (4):
  - the Joseph posterior equals independent null-space (R_warp = 0) or information-form conditioning to 4.4e-8 / 1.3e-10 correlation units (dense augmented Gaussian over all 51 local variables, oracle finite-difference H);
  - the IEKF fixed point holds to 3.9e-4 prior σ;
  - the source key is shared and the reference key persists and is reused;
  - the expiry ledger works;
  - backward processing works.
- `test_temporal_material_motion` (9):
  - textured scenes give full-rank information;
  - a **textureless aperture is rank-deficient (eigenvalue ratio 1e-32), invents no motion or velocity, and keeps the prior in-plane uncertainty**;
  - a photometric gain mismatch is flagged (predictive NIS per DOF ≥ 105), **but still moves static points by up to 31 mm in the fixture: there is no photometric model**;
  - the pyramid recovers 3 px and keeps static scenes static;
  - the resampling variance equals the exact bilinear remainder (u²);
  - end-to-end runner, journal and manifest work on CPU and CUDA;
  - CLI validation and forwarding work;
  - the global solvers are never called.

## 3. Full-cloud 3-frame pilots (actual inference, no GT)

Batch `temporal_rgbd_20261102_112822_3f5e9fd7` (material-following + legacy), compared with `temporal_rgbd_20261102_090025_a7b905eb` (local-correlated + simplified). All runs use the identical 95,967 / 34,369 original point IDs (checked), 8 cameras, `ideal` R_warp, measured sensor noise, forward direction, and every run is COMPLETE.

| | USB | Blade |
|---|---|---|
| material-following s/frame | 52.0 / 55.4 | 23.5 / 23.9 |
| peak CUDA allocation | 2.93 / 4.28 GB | 1.56 / 1.97 GB |
| rows | 150,182 / 159,141 | 50,805 / 50,848 |
| true image motion of moving points (median px) | 1.28 per frame | 1.45 per frame |

Post-hoc GT (`<batch>/posthoc_gt_four_modes.json`). Errors are in mm. The groups use **label-verified** points only (nearest-neighbour label and on-surface vote agree): USB 39,798 moving / 47,588 static, Blade 2,782 / 21,546.

| USB, frame 1 / 2 | moving median | moving p90 | moving p99 | moving < 3 mm | along-truth ratio q50 / q75 / q90 | static p90 / p99 |
|---|---|---|---|---|---|---|
| legacy | 15.6 / 29.8 | 22.5 / 44.9 | 33.6 / 75.5 | 5.9% / 1.6% | 0.00 / 0.23 / 0.85 ; 0.03 / 0.53 / 0.97 | 1.63 / 16.6 ; 6.05 / 36.4 |
| simplified | 16.6 / 33.2 | 22.3 / 44.5 | 22.7 / 45.6 | 6.0% / 1.4% | 0 / 0 / 0.04 ; 0 / 0 / 0.07 | 0 / 0 |
| local-correlated | 16.6 / 33.0 | 22.3 / 44.5 | 23.0 / 46.4 | 6.0% / 1.6% | 0 / 0 / 0.06 ; 0 / 0.02 / 0.10 | 0 / 0 |
| **material-following** | **14.2 / 25.5** | 22.5 / **48.2** | 49.3 / **122.6** | **9.9% / 4.3%** | 0.03 / 0.62 / 0.91 ; 0.29 / 0.75 / 1.02 | **0 / 0** |

| Blade, frame 1 / 2 | moving median | p90 | p99 | < 3 mm | along q50 / q75 / q90 | static p90 / p99 |
|---|---|---|---|---|---|---|
| legacy | 14.2 / 28.3 | 14.2 / 28.3 | 24.9 / 52.7 | 3.7% / 4.7% | 0 / 0 / 0.01 ; 0 / 0 / 0.55 | 0.008 / 10.5 ; 0.22 / 23.7 |
| simplified, local-correlated | 14.2 / 28.3 | 14.2 / 28.3–28.4 | 16.4–16.8 / 33.8–34.5 | 0% | ≈ 0 | 0 / 0 |
| **material-following** | 14.2 / 28.3 | 18.0 / 43.5 | 60.3 / 164.2 | 6.6% / 3.3% | 0 / 0.05 / 1.00 ; 0 / 0.38 / 1.00 | **0 / 0** |

**Material-following diagnostics** (label-verified moving points, frame 1 / frame 2):
- accepted max pixel shift, median: USB 0.45 / 1.45 px; Blade 0.13 / 0.96 px;
- rows with an accepted shift > 1 px: USB 34% / 60%; Blade 29% / 49%;
- current target pixel changed: USB 64% / 73%;
- assimilated: USB 90% / 91%; Blade 74% / 78%.

**Rejected full Gauss-Newton steps at frame 1, all points** (iteration 0 / iteration 1):

| Scene | No common MAP descent | Background projection | Identity support lost |
|---|---|---|---|
| USB | 2,214 / 11,870 | 3,339 / 3,951 | 2,671 / 3,312 |
| Blade | 619 / 2,108 | 194 / 220 | 1,099 / 1,268 |

**Identity bookkeeping, USB frame 1:** 74,142 same-surface collisions kept as identities; 1,226 + 158 + 0 hidden; 600 identities lost to occlusion and 1,517 gained in accepted steps.

**NEES** (full 3×3; moving / static median, frame 1 → frame 2):
- USB: 2.5 → 25.0 / 0;
- Blade: 1.9 → 200 / 0.

Moving points are overconfident at frame 2: `COVARIANCE_INCONSISTENT`.

## 4. Why moving points are not recovered

Measured with GT only (`tools/diagnose_material_observability.py`; frame 1 replayed exactly, 0.0 m from the published state; `<run>/posthoc_cause_diagnostic_frame1.json`).

| Frame 1 | USB | Blade |
|---|---|---|
| moving points with material rows at the true state | 30,637 | 2,131 |
| rank of position information at truth: 1 / 2 / 3 | **31% / 47% / 22%** | **36% / 40% / 24%** |
| prediction (no-motion) error, median | 15.8 mm | 14.2 mm |
| **oracle bound**: exact data, linearized at truth, same prior (q25 / median / q75) | 1.5 / **6.2** / 16.3 mm | 0.1 / **8.0** / 14.2 mm |
| actual material-following, same points (median) | 13.0 mm | 14.2 mm |
| IEKF **started at the GT state**, kept by acceptance / forced (median) | 8.7 / 7.3 mm | 14.2 / 11.3 mm |
| GT starts rejected vs prediction: no descent / background projection / identity lost | 10,883 / 3,450 / 2,275 | 1,809 / 176 / 421 |
| bad points (> 8 mm) whose objective is lower at the truth than at the accepted state | 30% | 25% |

There are three causes, in order of size:

1. **Observability (`IMAGE_MOTION_UNOBSERVABLE`).**
   - One material RGB-D sample per camera constrains essentially the surface-normal component. RGB adds almost nothing: USB RGB-only z-score 0.9, Blade 0.0.
   - About 76–78% of moving points have rank-1 or rank-2 information at the truth, so their tangential motion is unidentifiable. Even exact data and a perfect linearization leave a median 6–8 mm.
   - The estimator correctly keeps prior uncertainty there; this is the textureless-aperture behaviour the tests verify.
2. **The likelihood's optimum is displaced from the truth.**
   - Started at the truth, the IEKF drifts to 7–9 mm (USB). For 70–75% of badly estimated points the objective prefers the wrong state, and GT states are rejected for no descent against the prediction.
   - The audit attributes the residual at the truth to bilinear resampling of the rendered depth (about 1e-4 m, equal to the sampling error of exactly on-surface points). Against quantization-level declared noise (depth σ 8e-6·z on USB, 6.9e-8 m on Blade) and R_warp = 0, this makes the depth term overconfident by about ×2–3 in σ on USB and by orders of magnitude on Blade. The MAP therefore fits interpolation artefacts.
   - The data-derived resampling term reduces the GT-start drift only slightly (6.6 → 5.7 mm on the factor-8 replay).
   - No variance was invented or inflated to hide this.
3. **Association validity at the truth.**
   - The hard nearest-pixel mask rule rejects 3,450 true USB states (11% of moving points) as "projecting onto background" at silhouettes.
   - Losses of support validity reject another 2,275.
   - These are discrete visibility decisions acting on sub-pixel truths, not optimization failures.

Initialization (the basin) is a secondary factor:
- the actual error (13.0 mm) lies between the GT-start result (7–9 mm) and no motion (15.8 mm);
- the B1 pyramid and the resampling term changed the factor-8 frame-1 replay median by under 1 mm;
- any B2 correspondence initializer is bounded by the GT-started result. For that reason it was not implemented: the GT-started experiment is a strict upper bound for any initializer of this likelihood.

The spec forbids rigid groups, motion graphs, velocity constraints and invented noise, and they were not used. Recovery therefore needs a decision on one of these:
- a patch-level (multi-pixel) material measurement with its derived covariance, which assimilates tangential texture evidence;
- a calibrated resampling/warp discrepancy model;
- a different visibility rule at silhouettes.

## 5. Profiling (seeded-uniform factors, same flags)

| factor | scene | points | s/frame f1, f2 | peak CUDA GB f1, f2 | moving median mm f1, f2 | moving p90 mm f1, f2 | moving < 3 mm f1, f2 | static p99 mm f1, f2 | batch |
|---|---|---|---|---|---|---|---|---|---|
| 1 (gate) | USB_100109 | 95967 | 52.0 / 55.4 | 2.93 / 4.28 | 14.23 / 25.54 | 22.53 / 48.18 | 0.099 / 0.043 | 0.0 / 0.0 | `temporal_rgbd_20261102_112822_3f5e9fd7` |
| 1 (gate) | blade_103706 | 34369 | 23.5 / 23.9 | 1.56 / 1.97 | 14.16 / 28.31 | 17.98 / 43.49 | 0.066 / 0.033 | 0.0 / 0.0 | `temporal_rgbd_20261102_112822_3f5e9fd7` |
| 2 | USB_100109 | 47984 | 36.3 / 38.3 | 1.87 / 2.79 | 14.22 / 25.9 | 23.7 / 52.03 | 0.101 / 0.041 | 0.0 / 0.0 | `temporal_rgbd_20261102_113724_d101bace` |
| 2 | blade_103706 | 17185 | 18.6 / 17.7 | 1.2 / 1.46 | 14.16 / 28.31 | 20.34 / 50.37 | 0.049 / 0.026 | 0.0 / 0.0 | `temporal_rgbd_20261102_113724_d101bace` |
| 4 | USB_100109 | 23992 | 27.0 / 28.3 | 1.34 / 1.93 | 14.18 / 26.65 | 24.93 / 54.32 | 0.101 / 0.04 | 0.0 / 0.0 | `temporal_rgbd_20261102_113932_32f2cb14` |
| 4 | blade_103706 | 8593 | 16.3 / 13.9 | 0.96 / 1.18 | 14.16 / 28.31 | 21.05 / 55.8 | 0.043 / 0.016 | 0.0 / 0.0 | `temporal_rgbd_20261102_113932_32f2cb14` |
| 8 | USB_100109 | 11996 | 18.9 / 18.9 | 1.09 / 1.45 | 14.46 / 27.31 | 25.13 / 55.58 | 0.097 / 0.037 | 0.0 / 0.0 | `temporal_rgbd_20261102_114115_a481af4a` |
| 8 | blade_103706 | 4297 | 11.0 / 11.0 | 0.85 / 1.02 | 14.16 / 28.31 | 22.36 / 64.29 | 0.031 / 0.017 | 0.0 / 0.0 | `temporal_rgbd_20261102_114115_a481af4a` |
| 8 + pyramid 2 + second-difference | USB_100109 | 11996 | 23.3 / 23.6 | 1.16 / 1.56 | 13.46 / 23.76 | 28.4 / 56.45 | 0.112 / 0.058 | 0.0 / 0.004 | `temporal_rgbd_20261102_114230_ef7ecff7` |
| 8 + pyramid 2 + second-difference | blade_103706 | 4297 | 14.4 / 14.9 | 0.91 / 1.14 | 14.16 / 28.31 | 32.53 / 79.6 | 0.101 / 0.056 | 0.0 / 0.0 | `temporal_rgbd_20261102_114230_ef7ecff7` |

Notes on the table:
- Runtime and memory scale roughly linearly in points; the full cloud fits in 4.3 GB.
- Results are essentially independent of the subsampling factor, so the failure is not a sparsity artefact.
- The optional pyramid (2 levels) and resampling term improve the USB median by about 1–3.5 mm and the fraction under 3 mm by about 1.5 points. They also lengthen the tail (USB p90 25.1 → 28.4 mm at frame 1; Blade p90 22.4 → 32.5 mm) and cost about 25% more runtime. Neither recovers motion.

Different point counts change ownership and visibility, so factor ≠ 1 is not equivalent to the full cloud. The cuDSS global solver is never called (mock-tested).

## 6. HTML diagnostics

- Live viewers: `<batch>/<scene>_material-following/free_forward/temporal/index.html`.
- Full debug pages: `<batch>/<scene>_material-following/debug/index.html` (per-row q_t, pixel shift, reference/current support and identity in the traced-point records).
- GT audit: `material_warp_gt_audit/audit_20261102_104059_55e9f635/index.html`.

## 7. Exact commands (verified on this tree)

```bash
cd ~/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B -m unittest -v tests.test_temporal_material_warp tests.test_temporal_material_association \
    tests.test_temporal_material_local_covariance tests.test_temporal_material_motion
python -B tools/test_temporal_rgbd.py

# GT-only audit (writes a new audit_<stamp>_<id> child; never read by inference)
python -B tools/audit_temporal_dynamic_correspondence.py --scenes USB_100109,blade_103706 --frames 0,1,2 \
  --output-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/material_warp_gt_audit"

# full-cloud 3-frame gate (as run; add --temporal-material-pyramid-levels 2 / --temporal-material-resampling second-difference for the options)
bash tools/run_temporal_rgbd_critical.sh --mode material-following --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-warp-noise ideal --stop-after free --temporal-runtime-budget-seconds 600
python -B tools/run_status.py "$OUT" --watch 10        # OUT = the printed Results: directory

# post hoc (GT only, after inference)
python -B tools/evaluate_temporal_material_pilot.py --batch "$OUT" --batch "<batch with local-correlated/simplified>" --output "$OUT/posthoc_gt_four_modes_v2.json"
python -B tools/diagnose_material_observability.py --run "$OUT/USB_100109_material-following" --output "$OUT/USB_100109_material-following/cause_v2.json"
```

Do **not** run the 60-frame or ten-scene batch: moving-point recovery is not demonstrated.
