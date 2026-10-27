# Step 2A: Phase-1 grouped operator, one-step IEKF and covariance audit (read-only)

| Item | Value |
|---|---|
| Specification | `CLAUDE_STEP2_PHASE1_SUPERNODES_GPU_PERFORMANCE_AUDIT.md`, STEP 2A |
| Scene / run | `blade_103706`. Fixed inputs are the saved v2 production run `image_iekf_v2_rigorous_20261024` (34,369 points, 537 groups, 8 cameras) |
| Priors | Frame k uses `process6.predict` of the **saved** posterior of frame k−1, identical for every variant |
| Ground truth | Used only post hoc (`posthoc_*` fields, probes). Never inside a solve |
| Machine-readable | [phase1_grouped_operator_parity.json](phase1_grouped_operator_parity.json), plus [diagnostics/](diagnostics/): `phase1_objective_probe.json`, `phase1_slide_scan.json`, `phase1_kink_probe_frame1.json`, `covariance_blade_dense_*.json[l]` |
| Tools | `tools/audit_phase1_operators.py --parts A1 A2 A3`, `tools/diagnostics/{objective_probe,slide_scan,kink_probe,covariance_blade_dense}.py` |
| Code changed in this step | None. All sources and the Phase-1 arrays hash identical before and after (`unchanged_after_audit: true`) |

## A1: group basis G

| Check | Result |
|---|---|
| Groups | 537, no empty groups, 4 singletons. Size quantiles 1 / 2 / 70 / 107 / 115 |
| Centroid vs member mean | 7.8e-16 m |
| `expand` vs an independent implementation | 4.4e-16 abs on both CPU and CUDA |
| Adjoint `⟨Gz, y⟩ = ⟨z, Gᵀy⟩` | 3.8e-16 (CPU), 6.4e-16 (CUDA) |
| Retraction finite difference | 1.7e-9 |
| Shuffle invariance | exactly 0 |
| Pose Gram | 7 groups have rank < 6 (collinear or singleton members). Condition 5.2e3 to 4.2e5 |
| Ordering | group `[dt, dθ, du, dω]`; point `dx = dt + dθ×r`, `dv = du + dω×r` with `r = x_can − c_can`. Matches the pinned `GroupedObservation.expand` and `sparse_iekf._G` with the orientation block dropped |

## A2: reduced operator, gradient and solve

All blade checks are linearized at the saved state of frames 1 and 10.

| Check | Frame 1 | Frame 10 |
|---|---|---|
| H finite difference, per camera (relative) | 4e-8 to 8.5e-6 for 7 cameras; **1.4e-2 for camera 1** | 6.4e-7 to 1.1e-5 |
| H adjoint | 1.5e-15 | 2.9e-16 |
| `A_g` symmetry (λ = 1e-3 and λ = 0) | 1.9e-16 / 2.7e-16 | 3.3e-16 / 4.2e-16 |
| Min Rayleigh quotient of `A_g` on random vectors | 8.1e6 (positive) | 1.3e7 (positive) |
| `b_g` vs a finite difference of the true objective | 2.2e-7 (h=1e-4), 9.3e-8 (h=1e-5) | 8.2e-7, 6.5e-7 |
| Production Steihaug solve (30 PCG, η ≤ 1e-2) | relative residual 3.4e-2 (hit the iteration cap) | 8.1e-3 in 19 iterations (forcing met) |
| Accurate solve | 5.6e-10 after 2,000 iterations | 8.9e-13 after 540 iterations |
| Production vs accurate step | 90% relative difference | 68% |

The camera-1 outlier at frame 1 is a piecewise front tie. The objective is not differentiable there along the probed direction. It is not an implementation error, because every other camera and frame agrees to about 1e-6.

The synthetic dense oracles have 90 points, 6 groups and 2 to 4 cameras. On all three oracles, `A_g` and `b_g` match the explicit dense `Gᵀ(Λ⁻ + HᵀWH + λI)G` and `Gᵀ(HᵀWr − Λ⁻e)` to at most 3.4e-16. Cross-group coupling is 7–8% of the operator mass. The dense eigenvalues span 0.16 to 2.9e8, so the system is badly conditioned. That explains why 30 PCG iterations do not reach a small residual.

**Status: operator verified.** G, Gᵀ, H, Hᵀ, `A_g` and `b_g` are correct to round-off. They agree with the true objective's finite differences, apart from one piecewise point. The linear solve is inexact by design (a truncated Newton step), but A3 shows this does not cause the lag.

## A3: one-step IEKF on identical saved priors

Moving-part slide in mm, measured post hoc against the sparse GT part cloud. All variants share the same prior.

| Frame | True slide | Prior (predicted) | Production | Accurate linear solve | Budget 30 | Budget 100 + accurate | Prior σ_v (m/s) | Prior σ_x (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | 14.2 | 0.0 | 1.85 | 2.27 | 2.61 | 2.48 | 1.00 | 18.0 |
| 2 | 28.3 | 3.6 | 5.2 | 5.5 | 5.4 | 6.2 | 0.80 | 26.8 |
| 5 | 70.8 | 20.7 | 21.2 | 21.1 | 21.2 | 21.5 (converged) | 0.26 | 19.9 |
| 10 | 141.6 | 45.0 | 44.9 | 45.0 | 45.0 | 45.0 | 0.097 | 13.2 |
| 20 | 283.1 | 96.7 | 96.4 | 96.4 | 96.3 | 96.4 (converged) | 0.054 | 9.4 |

The variants are `production` = `IEKFConfig()`; `accurate linear solve` = η ≤ 1e-10 with 500 PCG; `budget 30` = `max_iters` 30; `budget 100 + accurate` = 100 iterations with the accurate solve.

**First frame of divergence from the high-accuracy solve: none.** Across every audited frame, production and the 100-iteration accurate solve differ by at most 0.63 mm in the moving slide. The extra budget lowers the energy (frame 1: 41,850 → 40,709), but it does not move the part. The lag is **not** caused by the iteration budget, the forcing tolerance or the PCG cap. No tuning of these parameters is justified, and none was made.

### Why the moving part lags

Two mechanisms, each measured.

**1. Frame 1–2: the local solve stalls in a non-smooth, non-quadratic region short of the image-preferred motion.**

* *Slide scan* (`phase1_slide_scan.json`): a rigid slide of the GT-moving points over the canonical static scene, scored with the image terms only. The minimum is at 9.9 mm at frame 1 (truth 14.2 mm) and 22.6 mm at frame 2 (truth 28.3 mm). At frame 10 the minimum is exactly at the truth, 141.6 mm. The observation model is therefore consistent with the motion.
* *Trust-region log* (frame 1): the first two steps hit the boundary with ρ = 1.45 and 1.23. Steps of 13–33 mm maximum point displacement then get ρ = 0.04–0.4. The radius shrinks to 0.25 and later to 6.1e-5. After that the iterations take near-zero steps (predicted decrease ≈ 0.1) while ‖b‖ stays at about 2e5. The quadratic model is only valid at sub-footprint scale.
* *Kink probe* (`phase1_kink_probe_frame1.json`): one-sided secants of the **true** objective along the solver's own next step, at the 100-iteration stalled iterate.
  * Left secant: equals the model slope `−bᵀd = −0.1632` for every step size h.
  * Right secant: −0.1617 at a 0.04 µm maximum point move, −0.137 at 0.39 µm, and −0.043 at 3.9 µm.
  * The objective is differentiable there (left and right agree at 4 nm), but its curvature changes discontinuously on one side, which points to a piecewise term (front, taper or coverage) switching on at sub-micron scale. At the 10-iteration production end the same test is perfectly smooth and linear (left = right = −0.513).
  * So the long solve walks into a near-kink and the trust region correctly refuses to step over it.
* *Objective probe* (`phase1_objective_probe.json`, all 3,293 moving points, frame 1): along the straight path from the production state to the true slide, the image energy **increases** (41,155 → 43,110). The non-rigid production state and the rigid true state lie in different basins, and the straight path between them goes uphill. A local solver started at zero slide therefore cannot be expected to reach the rigid optimum in one frame.

**2. Frame ≥ 5: the carried prior becomes overconfident, and the update adds nothing on top of the prediction.**

* The prior velocity σ falls from 1.0 m/s (frame 1) to 0.26 (frame 5), 0.097 (frame 10) and 0.054 m/s (frame 20). The true slide rate is 0.835 m/s.
* From frame 5 on, the posterior slide equals the predicted slide to within 0.5 mm in every variant. All the motion comes from the prediction; the image correction is about 0.5 mm per frame against 14 mm of true motion.
* *Objective probe at frame 10*: moving the part to the truth lowers the image energy by 3,430 nats (65,148 → 61,718). It raises the prior energy by 684,083 nats, almost all of it on the moving points (684,421 at α = 1 against 338 at α = 0). The prior, not the image, holds the part back.
* The carried covariance is the production default `rigeuler_block_information`, which is known to be overconfident. A4 explains why the exact alternative was not used.

**Status: moving-motion lag diagnosed.** It is (1) a non-convex, piecewise-smooth objective that stalls the per-frame local solve at 2–6 mm on the first frames instead of the image-preferred 10–23 mm, followed by (2) a velocity prior that the block-information posterior shrinks to σ_v ≈ 0.1 m/s while the truth is 0.84 m/s. This locks in the early under-estimate. It is not a solver-budget or linear-solve-accuracy problem.

## A4: covariance characterization (no swap made)

On a 1,000-point blade crop (15 groups), each method was compared every frame against a dense recursion with the same gain.

| Method | Result |
|---|---|
| Assumed-density marginals | Matches its own one-step dense Joseph to about 1e-8, but drifts from the true dense covariance: max variance 6.4e3 against a true 33 by frame 12. This is approximation error |
| Structured exact (`d + Gξ`), Jacobi equilibrated | Matches the full dense recursion to ≤ 1.3e-6 for frames 1–15. At frame 16 it jumps to a 0.70 relative error and stays at 0.06–4 after that |
| Conditioning at the break | min eig(B) falls from 3.5e-2 (frame 15) to 8.5e-4 (frame 16); cond(B) rises from 1.0e13 to 1.0e14, later up to 2.3e15. The equilibrated Cholesky and `pinv` disagree from that frame on |
| PSD repairs | 0 in every frame |
| True dense covariance | Max variance grows 10 → 219 (frame 15) → 2,938 (frame 31) → 21,346 (frame 40). Unobserved directions grow under the process noise. This is **growth**, separate from PD loss |

The structured exact covariance is algebraically correct. Its frame-16 departure is a float64 conditioning limit: it starts when cond(B) reaches 1e14, with no repair and no algebraic mismatch beforehand. **Unresolved:** the spec allows no swap without an isolated algebraic bug, and none was isolated. Production therefore still uses the overconfident `rigeuler_block_information`, which mechanism 2 above implicates. A sound covariance needs a better-conditioned parameterization of B, such as square-root or information form on the reduced block. That is a separate, scoped change and is not part of Step 2.

## Added after Step 2B-B1: sequence-level round-off sensitivity

A 1e-15 relative perturbation of the predicted state on the unchanged production path changes the forward trajectory by:

* frame 1: median 0.46 mm, maximum 12 mm;
* frame 9: median 2.4 mm, p90 12.5 mm, maximum 188 mm.

Source: `reports/diagnostics/phase1_roundoff_sensitivity.json`. The first divergent operation is the PCG solve truncated at 30 iterations on the badly conditioned reduced system (A2: eigenvalues 0.16 to 3e8). Coherence outcomes built on these trajectories, including the v2 192/192, are therefore not reproducible at round-off level; see [phase1_b1_small_contractions.md](phase1_b1_small_contractions.md).

## Status lines (Step 2A)

* **Operator: verified.**
* **Moving-motion lag: diagnosed.** Early-frame local stall in a piecewise objective, then an overconfident carried prior. It is not the solve budget.
* **Covariance: characterized, not resolved.** The exact structured form is correct until a float64 conditioning break at cond(B) ≈ 1e14; production uses an overconfident approximation.
