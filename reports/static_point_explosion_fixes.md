# Static-point explosion: what changed (2026-10-28)

This report answers `RGBDiekf_static_point_explosion_code_audit_2026-10-28.md` (audited commit 29a23a7).

Each fix has a policy switch, so the one-variable ablation the audit asks for (validation step 3) is available. Correction (same day, found by the root-cause trace, reports/free_point_runaway_root_cause.md): the first version of this fix also moved the `depth_front_only` front test onto the prior-linearized nu_D, so that switch did not reproduce the audited filter exactly. It again tests the iterate's own r_D, and the switches below now reproduce 29a23a7 bit for bit (storage F1, all points and frames). The default `rgbd_chi2` is unchanged. `--policy-override update_acceptance=final_iterate row_innovation_gate=depth_front_only final_visibility=frozen --sensor-noise config` reproduces the filter as audited.

## Reproduction (validation step 1)

The four-pass test fixture `static` reproduces the explosion with the audited code. It is a ray-cast scene with two plates, 6 frames at 30 fps, 4 cameras, and nothing moves:

- final static error: max 0.742 m;
- max speed: 4.44 m/s;
- 222 points off by more than 1 cm;
- joint: **REVOLUTE**. This was the one failing test of the suite (`test_static_scene_identifies_no_joint`).

Two traced runaways show the mechanism the audit describes:

| Point | Frame | Cameras | Converged | Prior NIS/dof | Accepted correction | Speed afterwards |
|---|---|---|---|---|---|---|
| 19807 | 1 | 1 | no (10 iterations) | 0.1 | 15 cm | 4.44 m/s, ballistic |
| 35198 | 2 | 1 | yes | 26 | 30 cm | 4.44 m/s, ballistic |

The prior NIS does not catch the first point. The cost at the final iterate does, through the prior term.

## Item by item

| # | Audit item | Change | Where |
|---|---|---|---|
| 1 | Non-converged iterate accepted | Acceptance no longer depends on convergence. It depends on statistical consistency, so consistent non-converged iterates (bilinear oscillation) are still accepted. Every point is **not** rejected for not converging; the ablation that did that is not repeated. | `estimator/point_kalman_cuda.py` `update_points` |
| 2 | Whole step accepted blindly | Every iterate s_j is checked against the **actual RGB-D samples at its own projection**: J = (s_j − s⁻)ᵀP⁻⁻¹(s_j − s⁻) + rᵀR⁻¹r, with J ≤ χ²₄ₘ(1−α). If the final iterate passes, it is accepted. Otherwise the last earlier consistent iterate is accepted after a fresh linearization. If none passes, the point is prediction-only (s⁺ = s⁻, P⁺ = P⁻). There is no renderer, no step clipping and no damping. | same; switch `update_acceptance` |
| 3 | NIS computed but only logged | The posterior cost (the IEKF form of the NIS, equal to r₀ᵀS⁻¹r₀ at a linear optimum) now drives the decision, with dof 4m. α = 2(1 − Φ(`visibility_n_sigma`)) = 0.0027, the tail of the 3σ rule already declared. No new number was added. | same |
| 4 | One-sided depth gate | The row gate is two-sided and is applied to the innovation **linearized back to the prior**, ν = r + H(x_j − x⁻), with S = H P⁻ₓₓ Hᵀ + R. ν_D > 3√S_DD gives MEASURED_SURFACE_IN_FRONT, as before. Any other row with νᵀS⁻¹ν > χ²₄(1−α) gives the new code 10, INNOVATION_INCONSISTENT. The test asserting that a point 1 m in front of its surface is VALID now asserts INNOVATION_INCONSISTENT; the old result is still asserted under `depth_front_only`. | `estimator/projective_terms.py`; switch `row_innovation_gate` (`rgbd_chi2` / `depth_two_sided` / `depth_front_only`) |
| 5 | 2×2 validity is not surface identity | The same row test checks colour and depth **jointly** against the prior uncertainty. A wrong surface of another colour, or a surface far behind the point, is rejected without temporal matching. A wrong surface with the same colour and depth stays indistinguishable; no local test can separate it. The 80° incidence / 3σ same-surface rule is unchanged: it still decides only whether the interpolant is valid. | same |
| 6 | Frozen pixel ownership at the iterate | At the acceptance linearization, a fresh SELF_OCCLUDED at the current states is added to the frozen assignment. A fresh DUPLICATE is not added, because it is reuse control and not visibility. During the iterations the assignment stays frozen, which is what kept the IEKF from oscillating. | `estimator/projective_visibility.py`; switch `final_visibility` |
| 7 | 0·NaN poisoning | The sampler marks any non-finite neighbour as INVALID_DEPTH, whatever `raw_valid` says, and returns finite placeholders for invalid rows. The free, rigid and joint `(q, q̇)` sums use `torch.where` on r and H. Tests: NaN/inf pixels in a sample, and NaN pixels next to a point in an update. | `projective_sensor.py`, `point_kalman_cuda.py`, `rigid_kalman_cuda.py` |
| 8 | Illustrative P_vv = I | `--velocity-prior-speed99 V` sets P_vv = (V/√χ²₃(0.99))² I = (V/3.368)² I, the audit's formula. V is **declared** by the user for the acquisition; it is not inferred from data or ground truth. No value is hardcoded, and the default is still the uncalibrated config. | `scene/noise_calibration.py` `velocity_prior`, `train.py` |
| 9 | q_c uncalibrated | Not changed. The audit notes that q_c cannot damp a velocity mean, and there is no acceleration data to calibrate it from. It is still set through `--noise-config`. | — |
| 10 | R uncalibrated, zero model variance | `--sensor-noise measured` is the new default. R is measured from the scene's own frames: the sensor part from MADs of consecutive-frame differences (depth fitted as a + b z²), the model mismatch from the cross-camera residuals of each endpoint cloud at its own instant, using a robust median moment match. No motion, no filter output and no ground truth enter it. | `scene/noise_calibration.py`, `noise_calibration.json` per run |
| 11 | Covariance after invalid association | The covariance comes from the accepted state's own rows. The gate uses the prior covariance with the prior-linearized innovation, so it does not become more permissive at later iterates. | items 2 and 4 |
| 12 | Diagnostics | New per-point arrays: `update_acceptance`, `posterior_cost`, `posterior_cost_dof`. Per frame: acceptance counts, cost/dof quantiles (accepted / rejected), accepted-but-not-converged count, and the **20 largest accepted corrections** with ids, Δv, rows, convergence and cost/dof, ready for `--debug-point-ids`. Traces add per-iterate cost and threshold, per-row NIS, the acceptance decision and the linearization state. | `free_projective_run.py`, `point_kalman_cuda.py` |
| 13 | Ballistic after loss | Not changed, as the audit asks: no attraction toward the start, no speed clipping. | — |

## Effect on the fixtures (ray-cast renders; ground truth used only after the run)

| | Static max error | Static speed max | Static joint | Prismatic moving error (median) | Prismatic joint |
|---|---|---|---|---|---|
| audited code | 742 mm | 4.44 m/s | REVOLUTE ✗ | 0.78 mm | PRISMATIC |
| new rules, config R | 35.7 mm | 0.21 m/s | STATIC | 0.79 mm | PRISMATIC |
| new rules, measured R | 47.1 mm | 0.28 m/s | UNIDENTIFIABLE | 0.83 mm | UNIDENTIFIABLE ⚠ |

The renders are noiseless, so the measured R collapses to quantization floors (σ_D ≈ 17 µm). With such an R, the joint adequacy test rejects a prismatic fit with a 3.6 µm residual. That test treats frames as independent and is hypersensitive with a near-zero R. It is a joint-stage limitation, outside this audit.

The end-to-end tests therefore run with the declared `--sensor-noise config` they were written for. Extra `*_measured` runs check the default path: no false joint on the static scene, free tracking kept on the moving one.

## Effect on a real scene (laptop episode 4, first 80 frames, free forward; no ground truth)

| Variant | Observed @ frame 79 | Displacement median / p99 / max | Speed max | Points > 1 m/s |
|---|---|---|---|---|
| A audited rules, config R | 18% | 284 mm / 5.4 m / 34.9 m | 11.05 | 1927 |
| B new rules, config R | 18% | 212 mm / 4.9 m / 11.4 m | 3.61 | 1338 |
| C new rules, measured R | 22% | 152 mm / 3.2 m / 12.5 m | 3.94 | 325 |
| E C + declared v99 = 1.0 m/s | 27% | 90 mm / 1.4 m / 3.2 m | 1.01 | 1 |
| D C + declared v99 = 0.5 m/s | 29% | 68 mm / 0.83 m / 1.9 m | 0.61 | 0 |

The measured R on this scene differs a lot from the config:

| | Measured | Config |
|---|---|---|
| RGB sensor σ | 0.001–0.004 | 0.05 |
| RGB model-mismatch σ | 0.016 | 0 |
| Depth sensor σ | 2.2 mm | 1 cm |
| Depth cross-camera mismatch σ | **2.4 cm** | 0 |

The depth mismatch is most likely extrinsic calibration error between the cameras.

The v₉₉ values above are a sensitivity check, not a recommendation; the speed must be declared per acquisition.

On real data, the remaining loss is points slowly leaving the foreground mask (MASK_EXCLUDED / OUTSIDE_IMAGE rows) and then extrapolating with small velocities. Item 13 forbids pulling them back, so this stays open.

Outputs: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/audit_fix_check_20261028_1228/`. `F_full_pipeline_20frames` there is a 20-frame full run (every stage, debug pages, page QA exit 0).

## Tests

Projective suite: `python -m unittest tests.test_point_kalman_cuda tests.test_projective_measurement tests.test_rigid_kalman_cuda tests.test_four_passes tests.test_joint_projective ...` passes. It includes the previously failing `test_static_scene_identifies_no_joint`.

New tests:
- row gate: both depth signs, colour, prior-linearized innovation, the three rules;
- non-finite pixels in a sample and next to a point in an update;
- acceptance: a 5 cm prior offset is rejected exactly (s⁺ = s⁻, P⁺ = P⁻) where the old rule accepted it and moved it further from the truth; consistent non-converged iterates are kept; the old rule is reproduced;
- static runaway bounds derived from each run's own velocity prior: speed < v₉₉ and displacement < v₉₉·T. The audited code broke both;
- `*_measured` / `*_v99` end-to-end runs.

Changed assertions: the "1 m in front stays VALID" assertion is now INNOVATION_INCONSISTENT (legacy still asserted). The Jacobian finite-difference test runs under `depth_front_only` because its colours are random by design. The missing-camera test uses the measured colour instead of a constant 0.5.

`tests.test_image_rigid_joint` (4 failures) and `tests.test_image_support` (1 failure) fail identically on a clean checkout of 29a23a7. They test the legacy rendered-splat pipeline, which these changes do not touch.
