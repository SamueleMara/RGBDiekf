# Image-gradient motion guidance and shared part motion — validation report (2026-11-02)

**Verdict.**
- The gradient velocity rows are the material-following linearization itself. `diagnostic` is bitwise identical to material-following on full-cloud USB and Blade; `replace` is equal to 1e-12 on fixtures. They add **no new information**, and deployable moving-point accuracy is unchanged (`DYNAMIC_NOT_RECOVERED`).
- **Shared part motion is the missing structural information.** With GT membership only (offline oracle), one twist solved from measured RGB-D recovers:
  - Blade's prismatic motion: member error 0.41 / 0.72 mm, against a 14.2 mm true displacement;
  - USB's revolute motion: 0.25 / 0.47 mm, against 16.8 mm;
  - while a true-static group stays at zero.
- No non-GT membership exists (`PART_MEMBERSHIP_UNAVAILABLE`), so no part-motion mode was added to inference.
- Static points are displaced by moving-surface rows the single-surface allowance accepts as their own surface. The first false impulse is identified exactly (§4).

Baseline is HEAD `9465cb1` plus the uncommitted local-correlated / material-following work, of which nothing is committed. Gate 0 tests: 129/129. Code map: `reports/gradient_motion_code_map.md`. Machine-readable: `reports/temporal_gradient_motion_validation.json`.

## 1. What was implemented (no existing estimator behaviour changed)

| Equation | Code | Test |
|---|---|---|
| u = π(X), J_π, J_Z = e₃ᵀR_cw (world, fixed camera) | `projective_camera.py:79` (reused) | `test_temporal_image_gradient_geometry` 1: 5 random and 8 actual Blade off-axis cameras, 3 depths; FD ≤ 1e-6, NumPy exact |
| RGB: ∂ₜI + ∇I·J_π·v = 0, G_I = ∇I·J_π | `temporal_motion_geometry.py:16 velocity_rows` | geometry 2: exact renders of an inclined moving plane, central time difference |
| Depth: (J_Z − ∇D·J_π)v = ∂ₜD, a ∥ normal | same; basis `:24 depth_basis` (diagnostic, never a state) | geometry 2–3: \|cos(a, n_true)\| > 1−1e-4, BᵀB = I, a·T ≈ 0; no row across a depth break or mask |
| Image derivatives with support validity and raw-pixel provenance | `temporal_image_gradients.py:34 gradients_at`, `:42 eulerian_difference` (Eulerian), `material_difference` | geometry 3–4: tangent RGB rank 1 / 2 / 0 for stripes / 2D texture / constant |
| Velocity chain rule G^v = dt·H (same measurement); Λ_v, b_v; identifiable directions vs the declared prior | `temporal_motion_observability.py:28 velocity_guidance` | observability 1: fusion equals an independent NumPy row sum to 1e-10; observability 2: textured = 3, aperture < 3, no invented in-plane velocity |
| Stage matrix (projected → foreground → frontmost → reference → support → admitted → accepted) | `temporal_motion_observability.py:56 stage_matrix` | observability 3: rear-surface points get no rows; 3 px crossings stay admitted |
| Photometric consistency (χ² at the declared α = 0.01, binomial) | `temporal_motion_observability.py:101` | geometry 6: 8% gain → `PHOTOMETRIC_MODEL_INVALID` |
| `replace`: same residual and Jacobian assembled from gradient rows | `temporal_gradient_iekf.py:31 gradient_prediction`, `GradientMaterialFrameUpdate` | `test_temporal_gradient_iekf`: parity with material-following ≤ 1e-12 including nonzero latent errors; filter equal ≤ 1e-12 on CPU and CUDA; same raw variables; occlusion kept |
| Pivot-centred twist chart X_t − o = Exp(dt·ξ̂)(X_r − o); ∂X/∂ξ = [I \| −[X_t − o]×]·dt·J_l(dt·ξ) | `temporal_part_twist.py:28 point_jacobian` (uses `se3.left_jacobian`) | `test_temporal_part_twist` 1: FD vs scipy expm at 0.34 rad; uncentred chart shown wrong |
| Prismatic (ω = 0) and revolute (v = v_o + ω × (X − o)) | same | part twist 2 |
| Group posterior P_ξ, Cov(X_i, X_j) = J_i·P_ξ·J_jᵀ | `:39 solve_group`, `:91 point_covariance` | part twist 3: equals a dense QR oracle; naive per-point replay is overconfident |
| Wrong merge → `GROUP_IDENTITY_INVALID` | `:97 membership_residual_test` | part twist 4: static drag > 1 mm and flagged |
| Group rows = the existing material residual, first-hit with all points | `:114 material_evaluator` | part twist 5: 3 px / 1.3 px / 0 group motion from RGB-D rows on CPU and CUDA |

**CLI.**
- New flag `--temporal-gradient-guide off|diagnostic|replace` (default `off`), for material-following only, in `train.py` and the batch runner.
- No part-motion flag and no GT option exist in `train.py` (tested).
- The oracle is the separate executable `tools/audit_rgbd_part_motion_oracle.py`.
- `--temporal-motion-report-point-ids` was not added: the existing `--debug-point-ids` / `--point-ids` already select the traced IDs.

**Tests.** `python -B tools/test_temporal_rgbd.py` gives **150/150** (129 previous + 21 new): 0 failures, 0 skips, CPU and CUDA, no source changes during the run (`tests/temporal_test_results.json`).

## 2. Gate C — 3-frame full-cloud pilots, deployable inference (no GT)

**`diagnostic`** batch `temporal_rgbd_20261102_173812_b92601bd`: `posterior_xyz`, `velocity`, `posterior_covariance`, `update_acceptance`, `camera_reason` and `material_rows` are **bitwise identical** to the material-following gate `temporal_rgbd_20261102_112822_3f5e9fd7` on both scenes. Moving and static errors are therefore exactly those of `reports/temporal_dynamic_point_recovery.md`:

| | USB | Blade |
|---|---|---|
| moving median error, f1 / f2 | 14.2 / 25.5 mm | 14.2 / 28.3 mm |
| static p99 | 0 | 0 |

The new diagnostics (frames 1 / 2), counting (camera, point) pairs at each stage:

| | Blade | USB |
|---|---|---|
| projected → foreground → frontmost | 274,952 → 268,559 → 267,837 | 767,736 → 755,972 → 754,596 |
| **with a usable temporal reference** | **57,010** | **160,122** |
| support valid = admitted → accepted | 50,805 → 50,154 | 150,182 → 146,952 |
| points with 1 / 2 / 3 data-identifiable velocity directions (data info > declared prior), f1 | 34% / 46% / 20% | 47% / 34% / 20% |
| depth row vs image normal, \|cos\| median | 1.000 | 1.000 |
| velocity guidance vs actual 6D update, cos median | 0.99999 | 0.99999 |
| photometric status | INVALID (2.1% / 2.8% exceed vs 1%) | INVALID (2.3% / 3.6%) |

The largest drop is the **reference stage**: in the full cloud about 8 points share a pixel, and only the first-hit winner commits a reference. The guidance diagnostic agrees with the actual update: it is the same measurement.

**`replace`**: batch `temporal_rgbd_20261102_175506_bd60e739` (full cloud) against material-following:
- Per evaluation it is equal to 1e-12, but it is **not trajectory-identical**. Round-off-level differences flip a few discrete accept/visibility decisions:

| Scene | f1 points > 1e-9 m | f2 points > 1e-9 m | f2 points > 1 mm | f2 acceptance flips |
|---|---:|---:|---:|---:|
| Blade (34,369 points) | 19 | 61 | 2 (max 177 mm) | 17 |
| USB (95,967 points) | 33 | 151 | 0 | 9 |

- All other points are identical, and medians are unchanged. It is the same physical model with the same information (`EQUIVALENT_REPARAMETERIZATION`). It also shows that the discrete visibility/acceptance decisions are sensitive to round-off for a handful of points.

**Cost** (factor 32: 1,075 Blade / 2,999 USB points; batches `…180157_3b0a9e6c` off, `…180300_d8ae346f` diagnostic, `…180609_2557c672` replace):
- off: 10–13 s/frame; replace: 10–15 s/frame. Peak 0.74–1.02 GB, identical across modes.
- Measured in-process, the diagnostic summary costs **0.24–0.31 s per frame**.
- The diagnostic batch's wall-clock 15–56 s/frame (and the full-cloud USB 113–166 s) is contaminated: another user's GPU job (Grounded-SAM2 masks, 2.1 GB) was running; a replayed step took 50–56 s against 13 s uncontended.
- No global solver is ever entered (mock test).

## 3. Gate D — offline oracle part motion (GT membership only; motion from RGB-D; twist starts at 0)

Run: `…/RGBDekf/part_motion_oracle/part_oracle_20261102_174451_7fd14477`.

- Members are label-verified points (nearest-neighbour label and on-surface vote agree): Blade 2,782 moving and 21,546 static; USB 39,798 moving and 47,588 static.
- 1→2 uses references re-committed at the **estimated** frame-1 positions.
- The free-point baseline (Arm P) is the material-following result on the same label-verified points: USB 14.2 / 25.5 mm, Blade 14.2 / 28.3 mm.

| Declared noise; frames 0→1 / 1→2 | Blade translation-only | Blade 6D twist | USB translation-only | USB 6D twist | true-static group, 6D |
|---|---|---|---|---|---|
| member error median, mm | **0.41 / 0.72** | 0.42 / 0.72 | 7.5 / 7.8 | **0.25 / 0.47** | Blade 0.01 / 0.02; USB 0.00 / 0.00 |
| member error p99, mm | 0.41 / 0.77 | 0.48 / 0.78 | 13.0 / 12.3 | 0.28 / 0.53 | — |
| true displacement, mm | 14.2 | 14.2 | 16.8 | 16.8 | 0 |
| ‖v_o − v_GT‖ (m/s) / ‖ω − ω_GT‖ (rad/s) | 0.024 / – | 0.024 / 0.020 | 0.18 / 1.0 | 0.014 / 0.010 | twist ≈ 0 |
| rows (0→1 / 1→2) | 3,534 / 2,118 | 3,539 / 2,116 | 60,609 / 35,693 | 60,599 / 35,742 | Blade 32,582; USB 76,810 (0→1) |

- The same-instant cross-camera noise (xcam) gives the same errors to ±0.02 mm.
- On identical support, J(ξ_GT) < J(0) and J(ξ_GT) < J(4 random ξ of equal norm) for both scenes and both pairs.

**Ablations** (6D twist, declared noise, 0→1):
- **Channels:** RGB-only does not recover the motion (Blade 14.2 mm, USB 16.8 mm, i.e. no motion). Depth-only is as good or better than RGB+D (USB 0.17 vs 0.25 mm; Blade 0.42 both).
- **Cameras:**
  - Blade: single cameras give 0.5–2.5 mm, except camera 0 (17 mm). The pair (0, 4) fails (14.2 mm: it sees only the in-plane slide); (1, 3, 5, 7) gives 0.40 mm; all 8 give 0.42 mm.
  - USB: single cameras give 0.27–0.60 mm, except camera 6 (16.8 mm). The pair (2, 6) fails; (0, 4) gives 0.11 mm.
- Which views carry motion depends on the motion direction relative to each camera, so camera choice matters even with correct grouping.

**Failures and limits of the oracle:**
- **Wrong merge** (moving + 10% static members): static members are dragged by a median of 10.3 mm (Blade) and 6.7–7.5 mm (USB). This is visible false static motion.
- **The residual membership test cannot discriminate.** It reports `GROUP_IDENTITY_INVALID` for the wrong merge, but also for the correct groups (45–93% of members failing under declared noise, 15–80% under xcam). The declared noise does not cover the interpolation discrepancy, so it is not a usable membership detector.
- **Covariance is uncalibrated.** Member NEES medians are 10⁵–10¹¹ (declared) and 10⁵ (xcam): σ_ξ is far too small.

## 4. Gate B — read-only audit of your 60-frame Blade run (`temporal_rgbd_20261102_140950_2d1ffaf3`)

Audit: `…/gradient_motion_audit_blade/gradient_audit_20261102_173700_a9337f9c`. The run directory was verified unmodified.

**Static point 27016** (static by both labels) receives a false velocity:

| Frame | Rows | What happens | Correction | Speed |
|---|---|---|---|---|
| 1 | 1 (camera 5, reference frame 0, pixel 190517) | first impulse | 7.3 mm | 0.43 m/s |
| 6 | cameras 6 and 7, material shifts −1.3, −3.4 / 9.0, −2.5 px | follows the wrong material | 83 mm | 6.0 m/s |
| 25 | — | — | 112 mm | — |

What happens at the frame-1 impulse:
- The row's RGB residual is 0.088–0.091, about 70σ of the declared RGB noise. Its depth residual is −1.1 mm.
- Re-measured in all 8 cameras at 0→1, only camera 5 changes at the point: Eulerian ∂ₜI ≈ 4.5/s, ∂ₜD ≈ −0.79 m/s, i.e. −13 mm per frame. The other seven show exactly 0.
- So the moving blade passes about 13 mm in front of the static handle point in camera 5. That gap is **inside the documented single-surface allowance** (z·tan80°·√2/f ≈ 1.4% of depth), so the blade's material is accepted as the handle's own surface.
- With the declared depth σ of 7.9e-8 m, this one row carries information of about 5.7e14 m⁻² and produces the whole correction. The velocity follows through the prior position–velocity coupling.

**Run-wide static drift** (NN labels): p99 0.0007 mm at f1, 0.89 at f2, 8.6 at f3, 46 at f7, 340 at f30, 748 mm at f59 (max 13 m). The moving error equals the full true displacement (832 mm at f59), so there is no recovery.

**Lag-3 audit.** At frame 3, about 22,000 static (camera, point) identities switch to `REFERENCE_OUTSIDE_LAG` and admitted rows fall from about 46,000 to 27,000. Identities of same-surface collision losers never re-commit, because only first-hit winners commit. Missing rows are never treated as zero velocity: process-only propagation is exact (test).

## 5. Answers to the four questions

1. **Is the temporal RGB-D change consistent with the true 3D velocity?**
   - Yes for depth on smooth, genuinely visible surfaces: the analytic test passes, \|cos(a, n)\| = 1, and the oracle depth-only arm recovers the motion.
   - It fails at moving-over-static overlaps within the single-surface allowance (§4), at depth breaks and silhouettes (no rows), for RGB under brightness change (`PHOTOMETRIC_MODEL_INVALID`), and on textureless in-plane slides (zero information).
2. **Does the multiview tangent/normal system give a correctly directed per-point velocity?**
   - It points in the same direction as the update the filter already makes (cos 0.99999), because it is the same measurement.
   - The tangent velocity is rank-deficient for 80% of points: only 20% have 3 identifiable directions.
3. **Can correct shared membership pool the information?** Yes:
   - Blade prismatic: 0.41 / 0.72 mm;
   - USB revolute: 0.25 / 0.47 mm;
   - the true-static group stays at zero.
   - The caveats are wrong merges, the non-discriminative membership test and the uncalibrated covariance.
4. **First false impulse:** frame 1, camera 5, pixel 190517 — the blade sliding 13 mm in front of a static handle point, accepted by the 80° incidence single-surface allowance, with quantization-level depth noise making one row decisive. The moving parts lack shared-part identity (membership), not image information.

## 6. Stop status and decisions

- **Gate E:** `PART_MEMBERSHIP_UNAVAILABLE`. The temporal runs have no coherence stage, and no non-GT membership file exists. No part-motion inference mode was added.
- **Gate F** (11 frames) and long sequences: **not run**, because the deployable 3-frame moving error did not improve.
- Decisions needed:
  - **(a)** a source of non-GT part membership, and a way to validate it (the residual test above cannot);
  - **(b)** whether the single-surface allowance (80° incidence) should be tightened or replaced for temporal identity, since it is the measured cause of the static impulses (a policy change, not done);
  - **(c)** whether to adopt the same-instant cross-camera depth discrepancy as declared noise;
  - **(d)** the reference re-commit rule for same-surface collision losers (lag expiry).

## 7. Commands (verified)

```bash
cd ~/Joint3DGS/RGBDekf && conda activate rigeuler3dgs
python -B -m unittest -v tests.test_temporal_image_gradient_geometry tests.test_temporal_image_gradient_observability \
    tests.test_temporal_gradient_iekf tests.test_temporal_part_twist
python -B tools/test_temporal_rgbd.py
python -B tools/audit_rgbd_gradient_motion.py --scene blade_103706 \
  --run-root '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_20261102_140950_2d1ffaf3/blade_103706_material-following' \
  --point-ids 14104,27016 --frames 0,1,2,3,4,5,6,7,23,24,25,26,30,59 \
  --output-root '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/gradient_motion_audit_blade'
python -B tools/audit_rgbd_part_motion_oracle.py --scenes blade_103706,USB_100109 \
  --output-root '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/part_motion_oracle'
python -B tools/run_temporal_rgbd_critical.py --mode material-following --direction forward --frame-limit 3 \
  --scenes blade_103706,USB_100109 --stop-after free --pcd-subsample-factor 1 --pcd-subsample-method seeded-uniform --pcd-subsample-seed 0 \
  --temporal-lag-frames 3 --sensor-noise measured --temporal-warp-noise ideal --temporal-gradient-guide diagnostic   # or replace
python -B tools/run_status.py "$OUT" --watch 10
```
