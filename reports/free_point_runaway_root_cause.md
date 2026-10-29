# Free-point run-aways: deterministic root cause (2026-10-28)

This answers `RGBDiekf_deterministic_root_cause_debug_spec.md`. It is an investigation only: no gate, acceptance rule, limit, damping or regularization was added or tuned. Every alternative below is an offline replay.

## Method

**Instrumentation (trace only, debug ids only; the solver path is unchanged):**
- `PointMeasurement.probe` (`estimator/point_kalman_cuda.py`) records, for every camera and every iterate:
  - visibility before and after the frozen assignment;
  - the owner and occupants of the nearest pixel;
  - the projection and Jacobians;
  - the raw 2×2 RGB, depth, mask and validity neighbours, with bilinear weights and interpolant gradients;
  - r, H_x, R, S_zz and the row NIS.
  Cameras that visibility excluded are recorded too.
- The iteration trace adds the step δᵀAδ, Ixx, gx and rᵀWr, and whether the point is still active.
- The final trace adds Λ⁻ and the probes at the final (and fallback) states.
- `run_free` adds s_prev, P_prev, F, Q, q_c and the signed dt to the trace.

**Independent oracle:** `tools/root_cause_oracle.py`, pure NumPy float64. Nothing is imported from `estimator/`. It re-derives:
- projection and the bilinear interpolant on the original images, cast to float32 as the GPU holds them;
- the same-surface rule and the row gates;
- A, b, δ, acceptance, P⁺ and Joseph;
- an exact additive split of each step into a prior pull plus per-camera, per-channel pushes: δ = A⁻¹[−Λe] + Σ A⁻¹[H_caᵀ r_ca / R_ca, 0];
- the iteration-map Jacobian ∂G/∂s, with G(s) = s + δ(s).

`tests/test_root_cause_oracle.py` holds the oracle to production on the ray-cast fixture, under the original and the current policies. Every iterate's state, A and δ, plus s⁺, P⁺, Joseph and the coupling identity, must agree.

**Driver:** `tools/root_cause_trace.py`.
- `find` ranks run-aways and their first divergent update.
- `analyze` writes one report per failure (`report.md`, float64 `analysis.json`, `arrays.npz`). Each report carries the git SHA and dirty-patch hash, source hashes, input-array hashes, cameras, policy and noise.
- `population` gives same-instant cross-camera residual statistics.

**Original-code reproduction.** These switches reproduce commit 29a23a7 bit for bit:

```
--sensor-noise config --policy-override update_acceptance=final_iterate row_innovation_gate=depth_front_only final_visibility=frozen
```

On storage_45135 F1, `posterior_xyz`, `velocity`, `point_status`, `camera_reason` and `valid_cameras` are identical to the archived 29a23a7 run: 0 differing elements over 346,437 points × 60 frames.

This needed one correction to my earlier audit fix. It had moved the `depth_front_only` front test onto the prior-linearized ν_D, so the legacy switch was **not** exact. `depth_front_only` now tests the iterate's own r_D again, as 29a23a7 did. The default `rgbd_chi2` gate is unchanged.

Outputs are in `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/root_cause_trace_20261028_1709/<scene>/root_cause/`.

## Traced failures (F1, original configuration)

The storage and USB ids are the spec's; the others are the worst `find` points.

| Scene / point | Frame | Source cam / colour | ‖δx‖ m | v⁺ m/s | J(s⁻)/dof | Prior penalty of s⁺ | Data cost, common rows, s⁻ → s⁺ | Driver | Same-instant check | ρ(∂G/∂s) > 1 | Cycle | Prior 1σ px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| storage 171031 | 1 | 3 / (0.118, 0.071, 0.071) | 0.231 | 13.36 | 127 | 182 | 1520 → 1844 | RGB 0.52, prior 0.43 | colour χ²/ch 47–293 | 10/10 | — | 2.8–2.9 |
| storage 170273 | 1 | 3 / same | 0.500 | 26.17 | 137 | 764 | 1639 → 2843 | RGB | colour χ²/ch 92–366 | 9/10 | 3 | 2.8–2.9 |
| storage 168322 | 1 | 3 / same | 0.566 | 32.65 | 153 | 1088 | 1221 → 1632 | RGB, prior | colour χ²/ch 43–407 | 10/10 | — | 2.9 |
| USB 81029 | 2 | 6 / grey 0.30 | 0.576 | 20.17 | 1000 | 1142 | 12030 → 1154 | depth 1.59 | depth r/σ 41–138 | 7/10 | — | 5.6–6.1 |
| USB 81820 | 1 | 6 / grey 0.30 | 0.364 | 19.82 | 560 | 422 | 4484 → 5842 | depth 4.34 | depth r/σ 7–167 | 7/10 | 3 | 3.2 |
| USB 19877 | 2 | 1 / (0.74, 0.09, 0.09) | 0.756 | 23.25 | **0.27** | 772 | **3 → 5736** | depth 2.31 | consistent where seen | 8/10 | 4 | 4.1–6.1 |
| stapler 126240 | 1 | 7 / (0.62, 0.05, 0.05) | 0.993 | 53.98 | 8710 | 3136 | (1 row) | depth 4.84 | consistent at frame 0 | 6/10 | 7 | 3.5 |
| stapler 125404 | 1 | 7 / (0.62, 0.06, 0.06) | 1.040 | 56.17 | 8340 | 3418 | (1 row) | depth 3.82 | consistent at frame 0 | 7/10 | 9 | 3.5 |

None of the 8 points converged, and all 8 were accepted as `ACCEPTED_FINAL_ITERATE`.

How to read the columns:
- **J(s⁻)/dof** is the data cost at the prior per degree of freedom: about 1 if R describes the rows.
- **Prior penalty of s⁺** is (s⁺−s⁻)ᵀΛ(s⁺−s⁻).
- **Driver** is the share of the summed exact step attribution along δx.
- **Same-instant check** looks at the canonical XYZ at the canonical frame in the other cameras.

## Hypotheses, proved or negated

| Spec hypothesis | Result | Evidence |
|---|---|---|
| Original solve ≠ independent oracle (algebra, sign, accumulator, covariance bug) | **Negated (8/8)** | At every production state: A, b, δ rel ≤ 1e-13; raw-pixel rows ≤ 1.3e-13; Joseph vs information ≤ 2.3e-14; finite-difference H of the same interpolant ≤ 5.1e-9 rel; F, Q, s⁻ exact, P⁻ rel ≤ 2.3e-18. |
| Velocity without the Gaussian coupling identity (F/Q/update bug) | **Negated (8/8)** | δv = P_vx P_xx⁻¹ δx to ≤ 1.4e-13 m/s. |
| Ballistic tail ≠ exact signed-dt propagation | **Negated** | x(t) = x_last + (t − t_last) v_last to ≤ 4e-14 m; P⁻ = F P Fᵀ + Q; s⁺ = s⁻ on every prediction-only frame. |
| Solve matches, but the objective increases or rows jump surfaces | **Confirmed (8/8)** | ρ(∂G/∂s) = 5–134 at most iterates. The objective increases at several iterates; Gauss–Newton residual error is up to 192σ on the same rows. 5 of the 6 points with common rows end with a higher data cost on them than at s⁻ (USB 81029 lowers it, by moving onto the wrong surface), and all 8 pay a prior penalty of 182–3418. |
| Objective decreases, and cross-view RGB on an edge dominates (invariant appearance wrong) | **Confirmed for storage (3/3)** | At the same instant and XYZ, all 7 other cameras disagree with the source pixel by 7–20σ per channel (source gradient 0.65–0.91 /px). Depth agrees in cameras 1 and 2 (\|r/σ\| < 1.5). The objective does not even decrease. |
| Depth dominates, and the sampled range lies on another surface | **Confirmed for USB 81029, 81820 and stapler (4/8)** | Admitted rows measure a surface 0.4–1.2 m **behind** the point (r_D/σ −69 to −187). The original one-sided rule keeps r_D < 0 as "motion". |
| Residuals small but P/R empirically inconsistent | **Partly** | See below. |

The population check at the endpoint shows what is wrong with R:
- the config R is right on average (mean r²/R ≈ 0.96 RGB, 0.92 depth on storage);
- the tails are far from Gaussian (excess kurtosis 36–1758; 5–7× the 3σ rate);
- within 2 px of the mask border, 18% of rows are beyond 3σ;
- 33–76% of the residual variance is shared by every row of the same point.

The run-away rows sit at χ² 10²–10⁴. No rescaling of a diagonal R explains them.

## The mathematics of the failure

Each run-away combines four things, all algebraically correct.

**1. A wrong or non-identifying measurement row at the prior.** Three cases, depending on the point:
- **Wrong surface.** The moving part leaves a pixel, and the camera now sees background behind the predicted point. The original rule admits negative depth innovations of any size (USB, stapler).
- **Wrong canonical colour.** One source pixel defines a fixed colour that the other views do not reproduce at the same instant (storage).
- **No bad row at all** (USB 19877, J/dof 0.27). Then only items 2–4 act.

**2. Gauss–Newton extrapolated far beyond its validity radius.** The bilinear interpolant's gradient is exact only inside one pixel cell, but the prior position 1σ is 2.8–6.1 px. That σ is dominated by the uncalibrated velocity prior: dt²σ_v² = 2.9e-4 m² against a source variance of about 4e-5 m².

The step therefore extrapolates a local gradient across several cells, edges and surfaces. The iteration map G(s) = s + δ(s) has ρ(∂G/∂s) ≈ ‖A⁻¹ Σ_a w_a r_a ∇²h_a‖ ≫ 1: large residuals times interpolant curvature (cross-terms and cell jumps) beat the information. It is locally expansive and cannot converge.

Two structural effects make the outcome arbitrary:
- **Zero-row iterates.** An iterate with no valid row gives δ = −(s − s⁻), an exact return to the prior. This closes **exact limit cycles** (periods 3, 4, 7, 9). The accepted state is whichever phase the iteration is in when `max_iekf_iters` = 10 stops it.
- **Cell-corner knife edge.** A static point at its forward prior projects exactly onto its own source pixel centre, a bilinear cell corner. There `floor(u)` (the 2×2 cell, its gradient, and every later iterate) is decided by round-off. The oracle and production agree to 1e-13 at every state, yet free-running replays of 168322, 170273 and 81820 end 0.35–28 m apart. The accepted state is not a stable function of the data.

**3. Acceptance without an objective check** (`final_iterate`, audit items 1–3). The non-converged, cost-increasing final iterate is accepted with its full covariance contraction. For 171031, the posterior cost goes from J(s⁻) = 1520 to J(s⁺) = 182 + 1844 = 2026.

**4. Velocity by exact Kalman coupling.** After one prediction from the endpoint (P_xv = 0):
- P⁻_xx = Σ₀ + dt²σ_v² I + dt³/3 q_c
- P⁻_vx = dt σ_v² I + dt²/2 q_c

Any position-only correction therefore yields δv = P_vx P_xx⁻¹ δx. With σ_v = 1 m/s and dt = 1/59 s, the gain is 30–58 s⁻¹, that is 0.5–1.0 × 1/dt: most of a position correction is explained as velocity. For example, 0.23 m becomes 13.36 m/s.

The point then leaves its pixel. At the next frame its cameras report MASK_EXCLUDED or SELF_OCCLUDED, and it propagates ballistically and exactly (the expected behaviour, audit item 13).

## Offline one-variable replays (never used by the filter)

Each variant runs from the same s⁻ and P⁻ with the same pixels; per-iterate trajectories are in the reports.

**Channels:**

| Variant | Results |
|---|---|
| RGB only | ≤ 0.02 m everywhere except storage 168322 (0.155 m), whose RGB rows alone run away |
| Depth only | storage < 3 mm; USB 81029 still 0.39 m (the wrong-surface depth row alone) |

**Leaving out one camera** (a diagnostic, not a gate) removes the run-away when the dropped row is the wrong one:

| Scene | Dropped camera | Effect |
|---|---|---|
| stapler | camera 6 (the background row) | 1.0 m → 0 |
| USB 81820 | camera 7 | 0.37 m → 8 mm |
| storage 171031 | source camera 3 | 13.4 → 2.1 m/s |
| storage 171031 | camera 2 | 13.4 → 22.9 m/s (worse) |

**Fixed first-iterate row set:** identical to the original in all 8 cases. The row set changes are consequences of the bad steps, not their cause.

**Measured R** (Codex calibration; the endpoint covariance is not recomputed):

| Scene / point | Original | Measured R |
|---|---|---|
| storage 171031 | 13.4 m/s | 7.7 m/s |
| storage 170273 | 26.2 m/s | 36.9 m/s |
| USB 81029 | 20.2 m/s | 21.3 m/s |
| other five | — | ≤ 1.8 m/s |

It does not remove the mechanism. On these noiseless renders the measured R is tiny (RGB σ ≈ 5e-3, depth ≈ 4 mm), so wrong rows weigh even more.

**P_vv from a declared v₉₉** (chain replay from the endpoint):

| Effect | Details |
|---|---|
| Velocity gain falls | 52 → 24–37 s⁻¹ (v₉₉ = 0.5) |
| Prior σ in pixels shrinks | — |
| Outcomes still erratic | stapler 125404: 1.04 m → 0.75 m, 6.5 m/s at v₉₉ = 0.5; storage 170273: 0.85 m/s at 0.5, 0 at 1.0 |

The run-away is reduced but not removed, and it stays non-monotonic in v₉₉: a phase-of-cycle effect.

## Conclusion (uncertainty-qualified)

Within the eight traced first-divergence events (three synthetic scenes, F1), there is **no implementation error**. Algebra, sampling, covariance, prediction and ballistic propagation all match an independent oracle to round-off.

The run-aways are correct Kalman algebra applied to:
- **(i) a measurement model that is wrong for those rows:**
  - surface identity: a background surface admitted by the one-sided depth rule;
  - invariant single-pixel appearance: a canonical colour inconsistent with the other views at the same instant;
- **(ii) a Gauss–Newton iteration run far outside the one-pixel validity radius of the bilinear interpolant.** It is locally expansive (ρ ≫ 1), with exact limit cycles through s⁻ and a round-off-decided cell at the source pixel corner. Its arbitrary final phase is accepted without an objective check;
- **(iii) a prior whose uncalibrated P_vv** both widens that search radius to 3–6 px and converts most of any position correction into velocity, δv = P_vx P_xx⁻¹ δx with gain ≈ 0.5–1.0/dt.

Point (ii) explains USB 19877, which has no inconsistent measurement at all.

**Not established:**
- the physical cause of the dark storage feature: it is local, since camera 3 is unremarkable as a source in the population;
- B1, the real scenes and points beyond these eight;
- whether the current default rules (`rgbd_chi2`, `last_consistent_iterate`) neutralize each mechanism on these exact points. That needs the same trace under the current policy.

The chain replays hold the other points at their production states, which affects the frozen pixel assignment. The classification cut-offs in the report (3σ, χ²/channel > 9, ρ > 1) only label evidence.

## Reproduce

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
OUT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/root_cause_trace_$(date +%Y%m%d_%H%M)"; mkdir -p "$OUT"
SC="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/storage_45135"
~/miniconda3/envs/rigeuler3dgs/bin/python train.py --scene "$SC" --output "$OUT/storage_45135" --direction forward --stop-after free \
  --pcd-subsample-factor 1 --covariance-precision float64 --debug none --no-evaluate --no-qa --sensor-noise config \
  --debug-frames $(seq -s, 1 59) --debug-point-ids 171031,170273,168322 \
  --policy-override update_acceptance=final_iterate row_innovation_gate=depth_front_only final_visibility=frozen
~/miniconda3/envs/rigeuler3dgs/bin/python tools/root_cause_trace.py find "$OUT/storage_45135" --top 10
~/miniconda3/envs/rigeuler3dgs/bin/python tools/root_cause_trace.py analyze "$OUT/storage_45135" --point 171031 --alt-noise measured --alt-speed99 0.5 1.0
~/miniconda3/envs/rigeuler3dgs/bin/python tools/root_cause_trace.py population "$OUT/storage_45135"
```
