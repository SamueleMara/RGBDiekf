# Rendered binary mask as an IEKF image measurement (2026-10-29)

This report answers `RGBDiekf_Codex_Rendered_Binary_Mask_IEKF_Replacement.md`. Nothing is committed: base commit `9a9f16f`, plus the uncommitted worktree changes.

Source hashes (first 16 hex digits):

| File | sha256 |
|---|---|
| `estimator/rendered_mask_measurement.py` | `32e2ef50f75574a5` |
| `estimator/rendered_mask_iekf.py` | `e3efe044217482f6` |
| `estimator/free_projective_run.py` | `110b3988fb7c5e61` |
| `estimator/point_kalman_cuda.py` | `2f7382ca19008ec3` |
| `train.py` | `49b70f61622a3dd9` |
| `configs/projective_rgbd.json` | `3a1ce8cdfebbe176` |

Each run's `run_manifest.json` records the full hashes.

**Bottom line.**
- **Implemented:** the measurement exactly as specified, with tests, and the old hard-geometry path removed.
- **Not working:** in an A/B on storage, USB and stapler, assimilating it **degrades** the free trajectories, and on the static fixture it creates a false joint.
- **Limitation:** the mechanism is a property of the specified measurement (§9 Limitations). It is not hidden or patched.
- **Mode:** `off` stays the default.

## 1. Changes

**New modules:**
- `estimator/rendered_mask_measurement.py`: the renderers, rows and matrix-free J.
- `estimator/rendered_mask_iekf.py`: the joint IEKF, PCG, line search and covariance.

**New tests:**
- `tests/test_rendered_mask_measurement.py` (A, B)
- `tests/test_rendered_mask_iekf.py` (C, D, E)
- `tests/test_rendered_mask_four_pass.py` (E, F end to end)

**New config:** `configs/example_mask_image_noise.json`.

**Modified:**
- `estimator/free_projective_run.py`: off / diagnostic / on, new arrays, statistics, debug images.
- `estimator/point_kalman_cuda.py`: the hard-geometry hooks removed. The trace probes stay.
- `train.py`: `--rendered-mask-measurement`, `--mask-image-noise-config`, removed flags, source hashes, resume key `mask_noise`.
- `configs/projective_rgbd.json`: the `rendered_mask_measurement` section.
- `README.md`.

**Kept as they were:** the half-pixel synthetic intrinsics fix (`scene/dataloader.py`), the GT evaluator fixes (washer labels, continuous joint), and the exact legacy `depth_front_only` switch.

## 2. The measurement

`y_obs` is the measured binary mask of the frame (camera c, frame k, 1 = white foreground), as loaded from the dataset.

`y_hat` is the pseudomask rendered by drawing **every** transported point white on black in every camera. That includes occluded points, points that are SELF_OCCLUDED or DUPLICATE for RGB-D, and points whose RGB-D rows were rejected.

The innovation `r = y_obs − y_hat` is assimilated **only on pixels where `y_obs = 0`**, so `r = −y_hat` there. White predicted inside the measured mask is never penalized. The mask is used once per frame, with a fixed prior.

## 3. Renderers

**Hard binary pseudomask B** (audit and debug images):
- Each point is drawn at its nearest pixel centre, `torch.round` (half to even), with the same convention as `projective_visibility`.
- B is the OR over points.
- A point behind the camera or outside the image renders nothing in that camera.

**Differentiable pseudomask m̂** (the IEKF linearization):
- Each point splats bilinearly onto the four centres of its floor cell.
- m̂ = 1 − Π(1 − a), with the exact derivative Π_{j≠i}(1 − a_j) · ∂a_i/∂(u,v) · J_u.
- The exclusive product uses a log-sum of the factors below 1 plus a count of the factors equal to 1, so there is no division by 0.
- At integer coordinates the convention is the right derivative of the floor cell.

**Tests (A, B):**
- r = −1 exactly for a point on a background pixel centre.
- No row on foreground.
- Two points in one pixel are white once.
- Occluded points render.
- No black padding outside the field of view.
- The Jacobian matches finite differences with overlapping contributors (1–3 cameras).
- Saturated and coincident points are finite and deterministic.
- J and Jᵀ are adjoint.

**Mismatch between the two:** m̂ is fractional away from pixel centres, so the hard count and the anti-aliased objective differ. Both are reported.

## 4. Equations and solver

With S = P⁻_xx, L = P⁻_vx S⁻¹ and C = P⁻_vv − L P⁻_xv, the update is exactly a 3N position problem; v = v⁻ + L(x − x⁻) is the prior-conditional velocity:

```
A δX = b,   A = blockdiag(S⁻¹ + I_rgbd) + Jᵀ R⁻¹ J,   b = −S⁻¹(X − X⁻) + g_rgbd + Jᵀ R⁻¹ r
```

- I_rgbd and g_rgbd are the unchanged `PointMeasurement` terms.
- Rows exist only at measured-background pixels touched by a footprint. Omitting the other pixels is exact.

**Coupling:** points are grouped into components through shared rows by label propagation.
- A point with no row gets an exact 3×3 solve.
- Coupled components get a matrix-free PCG: 3×3 block-Jacobi preconditioner, per-component step lengths, local indices.
- Cross-point blocks Jᵢᵀ R⁻¹ Jⱼ are kept. The test fails if they are dropped.

**Stopping:** each component stops at max(1e-2 × the IEKF step tolerance, (10⁻³)² × the initial energy), an inexact Newton step.

**Determinism:** all reductions are deterministic. They are sorted segment sums, plus scatter min/max, which don't depend on order.

**Globalization:** per-component backtracking (α = 1 … 2⁻¹⁰) on the actual objective ½|X − X⁻|²_{S⁻¹} + ½Σ r²_rgbd/R + ½Σ m̂²/R_mask. Each trial is compared on the **same** measurement set:
- RGB-D rows sampleable at both states;
- mask pixels the component touches at either state.

Without this, a 2×2 support flipping at a pixel-centre knife edge, or a pixel newly shared with another component, made every step look like an increase.

**Acceptance:**
- Mask-coupled points keep the final iterate of the monotone joint iteration.
- Points the mask never touched take the plain IEKF step and the policy's own rule (`final_iterate` / `last_consistent_iterate`), exactly as `update_points` does.

**Tests (C):**
- `Jv`, `Jᵀw`, A·v and the PCG step match a dense image-by-position oracle, overlapping case included.
- The 3D-reduced step and Δv = P_vx S⁻¹ Δx equal a full dense 6N IEKF to 1e-8.
- The fast trial objective equals a full re-render to 1e-10.

## 5. Removed hard-geometry path

The snapshot is in `attic/hard_geometry_snapshot_20261029/`: the 567-line patch, copies of the files, and SHA256SUMS.

**Deleted:**
- modules `estimator/constrained_point_update.py`, `estimator/truncated_position_moments.py`, `estimator/visual_hull_constraints.py`;
- tool `tools/geometry_probe.py`;
- tests for those three modules and `tests/test_geometry_four_pass.py`.

**Removed from production:**
- the `GeometryUpdate` hooks in `update_points`;
- the `geometric_constraints` policy section;
- the `geometry_*` memmaps and traces;
- `STAGE_SOURCES` entries.

**Removed flags:** `--geometry-constraints`, `--geometry-depth-mode` and `--geometry-posterior-mean` now stop at argument parsing with an explanation. They are never aliased to the new measurement (tested).

A search of the code finds no remaining reference except those two guards.

## 6. Covariance (named approximation `pixel_marginal_adf`)

R_eff[p,i] = R[p] + Σ_{j≠i} J[p,j] S_j J[p,j]ᵀ and S_i⁺ = (S_i⁻¹ + I_rgbd,i + Σ_p J[p,i]ᵀ R_eff⁻¹ J[p,i])⁻¹. The 6D block is reconstructed as [[S⁺, S⁺Lᵀ], [L S⁺, C + L S⁺ Lᵀ]].

**Against the exact dense joint inverse** (test D):
- **Without shared pixels:** identical (rel 1e-9).
- **With shared pixels:** variance ratio ADF/exact is 1.10–63 (median 1.58), so conservative on that fixture. No conservative bound is claimed.
- **Two points in one pixel:** they have non-zero exact cross-covariance, which ADF ignores.

All published blocks were SPD. A point with no row and no motion keeps (s⁻, P⁻) exactly.

This is not an exact Bernoulli posterior.

## 7. Parity, real scenes, ground truth

**Parity** (`tests/test_rendered_mask_four_pass.py`, ray-cast fixtures, both directions):
- `off` writes no new file.
- `diagnostic` changes no state, covariance, status or acceptance array (bit-identical).
- `on` keeps the moving part tracked (criterion of the unconstrained test) with signed dt.
- **But `on` FAILS the static scene:** forward identifies a false PRISMATIC joint. Details are in §9.

**Real scenes**, A/B on corrected intrinsics: forward, free stage, `--sensor-noise config`, mask variance 1/12 (UNCALIBRATED example), ground truth used only afterwards. Outputs are in `results_11_10_26/RGBDekf/rendered_mask_20261029_1107/` (`ab_forward_summary.json`).

| scene | mode | s/frame | static median | static p99 | static > 10 cm | moving median (true motion) | max speed |
|---|---|---|---|---|---|---|---|
| storage | off | 1.5 | 10.1 mm | 1.40 m | 23.4% | 167 mm (532) | 4.3 |
| storage | on | 5.5 | 12.6 mm | 1.75 m | 26.0% | 167 mm | 15.7 |
| USB | off | 0.5 | 0.1 mm | 1.19 m | 12.6% | 586 mm (899) | 4.6 |
| USB | on | 7.5 | **91 mm** | 3.37 m | **49.1%** | 552 mm | 27.9 |
| stapler | off | 0.7 | 16 mm | 1.25 m | 24.1% | 1589 mm (1659) | 5.1 |
| stapler | on | 11.6 | 66 mm | 2.80 m | 42.8% | 2130 mm | 27.9 |

Across all frames, the update removed only 2–3% of the outside-white pixels:

| Scene | Before update (s⁻) | After update (s⁺) |
|---|---|---|
| storage | 4.33M | 4.19M |
| USB | 5.55M | 5.44M |
| stapler | 12.6M | 12.3M |

At the last frame, 26k, 38k and 58k points were in uniform background (no boundary in any footprint).

The `off` baseline itself drifts: 12–24% of static points end more than 10 cm off. That is the free filter's existing behaviour, separate from this contract.

Traces of the spec's points (`storage:171031`, `USB:81029`, `stapler:126240`) are in `on_fwd_*/free_forward/numeric/trace_frame_*.json`. Debug images (M_obs | B(s⁻) | B(s⁻) & ~M red | B(s⁺) & ~M red | m̂ | selected residual | points per pixel) are in `on_fwd_*/free_forward/mask_debug/` for frames 1, 2, 30 and 59.

Downstream (coherence → F2/B2 → joint → support) and the backward direction were not run on these scenes; the static-fixture failure and the forward A/B came first.

## 8. Runtime

Storage, 6 frames, forward:

| points | off s/frame | on s/frame | GPU peak off / on | PCG iterations (median) |
|---|---|---|---|---|
| 28,870 | 1.71 | 3.89 | 328 / 324 MB | 12 |
| 115,479 | 2.13 | 4.65 | 819 / 756 MB | 21 |
| 346,437 | 2.82 | 6.85 | 1884 / 1996 MB | 41 |

The former hard QP/EP mode took 30–64 s per frame on full storage.

**Over the full 60-frame A/B** (§7), `on` cost 5.5–11.6 s per frame, 3.7–17× the `off` time; the stapler is the slowest.

**Where the time goes** (full storage, per frame):

| Part | Time |
|---|---|
| RGB-D terms | ~0.8–2.9 s |
| Rendering | ~0.8–1.2 s |
| PCG | ~0.9 s |
| Line search | ~1.7 s |
| Diagnostics | ~0.5 s |

The development iterations are kept and labelled `_superseded_*`, so the history of the profile numbers is visible.

## 9. Limitations (measured)

1. **True rim points have a non-zero residual.** A surface point on the silhouette rim lies within half a pixel of the rasterized mask edge, so its bilinear footprint always spills partial coverage onto the neighbouring background pixels. Each frame pushes it, and through L ≈ 0.88/dt (the uncalibrated 1 (m/s)² velocity prior) the push becomes velocity.

   On the static fixture, the 4% of points with mask rows end with a median displacement of 8.6 mm and a median speed of 0.054 m/s, against 0.36 mm and 0.001 m/s for the other points. The p99 displacement is 70 mm against 8 mm with `off`, the maximum is 635 mm, and the result is a false PRISMATIC joint.

   This is the specified measurement (one-pixel splat, one-sided rows), not a coding error. Fixing it would mean changing the footprint model, the mask variance, or the observation model, which is your decision.
2. **Uniform background (§3.4).** A footprint entirely in black has a non-zero but only sub-pixel-periodic gradient. In the fixture test, a point 30 cm outside moved at most 2.2 mm, so no attraction is invented. These points are reported as `MASK_ROWS_UNIFORM_BACKGROUND`, and there are tens of thousands of them on the real scenes.
3. **Shared-pixel covariance:** ADF is conservative on the fixtures but its error is unbounded (variance ratio up to 63).
4. **R_mask is an UNCALIBRATED modelling choice** (1/12 for a uniform boundary coverage). The masks are segmentations of the same RGB, so R = blockdiag(R_rgbd, R_mask) is a composite-likelihood approximation.
5. **Acceptance on mask-coupled points:** these points use the joint monotone acceptance, not the RGB-D χ² rule; their RGB-D consistency is reported separately (`rendered_mask_rgbd_consistent`).

## 10. Not completed

- **One test fails:** `test_on_keeps_a_static_scene_static` (§9.1). It was not weakened.
- **Not run on the named scenes:** downstream stages and the backward direction (§7). Both directions are covered on the fixtures.
- **Debug pages:** the mask diagnostics exist as PNG images and numbers, but the HTML debug pages don't show them yet.
- **Production default:** `on` must not become the default. The validation failed.
