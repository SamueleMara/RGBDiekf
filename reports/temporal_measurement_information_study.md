# Measurement-information and likelihood-falsification study — USB_100109 and blade_103706, frames 0→1→2

**Conclusion.**
- The images determine the **surface-normal component** of each point's motion.
- They determine **one tangential component only where there is texture or depth curvature** (about half of USB's moving points, almost none on Blade).
- The **second tangential component is unobservable** from single-pixel material observations.
- Multi-pixel patches make all three directions informative. However, **no patch motion model tested is physically correct**: the rigid image patch and the 3D-translation patch deviate from the GT per-pixel motion by ≥ 0.1 px in about half of USB's 5×5 patches, and Blade patches mix in the static handle.
- Their likelihoods are biased: per-point minima lie off the truth, and the MAP started from the truth drifts 4.5–9 mm on USB.

By the agreed criterion, **no new observation was implemented**. No estimator was modified.

**Provenance.**
- Tool: `tools/study_material_measurement_information.py` (GT only, offline).
- Runs:
  - `…/RGBDekf/measurement_information_study/study_20261102_132941_f123ad77` (main);
  - `…/study_20261102_135141_c79ff4f7` (3D-translation patch check).
- Full tables: `reports/temporal_measurement_information_study_tables.md`. Machine-readable: `reports/temporal_measurement_information_study.json`. Curve figures: `<main>/{USB_100109,blade_103706}_curves.png`.

**Setup.**
- Points: 2,000 moving and 800 static per scene, sampled with a fixed seed (0) from label-verified points. Labels are verified when the nearest-neighbour label and the multi-frame on-surface vote agree. Original IDs are stored.
- References: true camera-local references, i.e. the GT first-hit pixel at τ on the point's own measured surface.
- Point states are independent: no rigidity, no neighbour graph. GT is used only to place states and to score.
- Prior for the MAP tests: the isotropic zero-motion prediction, σ_p = σ_v·dt = 16.9 mm (the run configuration).

**Observation models.**
- single pixel (`rgbd-1`, the current material-following row);
- k×k RGB-D patches with k = 3, 5, 7, 9 (`rgbd-k`);
- depth-only (`depth-1`, `depth-5`);
- gain-normalized RGB plus depth (`gainnorm-5`, `gainnorm-9`).

The current-sample covariance is the exact shared-raw-pixel form W·diag(σ²)·Wᵀ. Patch membership requires a valid foreground reference pixel on the centre's surface and a complete one-surface 2×2 current support. Background is never sampled.

## 1. Uncertainty sources

| | USB | Blade |
|---|---|---|
| declared RGB σ (= quantization 1/255/√12) | 1.13e-3 | 1.13e-3 |
| declared depth σ at 1 m | 8.3e-6 m | 6.9e-8 m (depth quantum 2.4e-7 m) |
| **same-instant cross-camera discrepancy, no GT** (other cameras sampled at the measured backprojection; 68.27% quantile), depth | **1.13e-4 m** | **6.9e-5 m** |
| same, RGB (includes view-dependent shading) | 2.8–4.4e-3 | 2.3–4.0e-3 |
| temporal residual at the truth, single pixel, moving: depth / RGB | 1.6e-4 m / 1.1e-3 | 2.6–2.8e-4 m / 2.0–3.0e-3 |
| temporal residual with each patch pixel's **own** true motion (k = 3, 5, 9), depth | 1.5–1.7e-4 m | 0.8–1.4e-4 m |
| temporal residual of the **rigid patch** at the truth, depth, k = 3 / 5 / 9 | 2.5 / 3.5 / 5.6e-4 m | 1.7 / 2.8 / 5.0e-4 m |

**What this shows.**
- RGB on USB is at quantization level, so the declared RGB noise is correct.
- Declared depth noise is too small by about 14× on USB and about 1000× on Blade, compared with a discrepancy measured without GT at a single instant (interpolation and rasterization).
- On USB that same-instant term explains about 70% of the temporal depth residual σ, and less on Blade.
- The rigid-patch approximation adds a model error that grows with patch size, while exact per-pixel motion does not.
- Real frames show no brightness change: static residuals are exactly 0.

## 2. Which motion directions are recoverable (frames 0→1, same-instant discrepancy added)

Recoverable = data-only standard deviation below one third of the true displacement.

| | USB 1-px RGB-D | USB 1-px depth | USB 5×5 | Blade 1-px RGB-D | Blade 1-px depth | Blade 5×5 |
|---|---|---|---|---|---|---|
| moving points with 0 / 1 / 2 / 3 recoverable directions | 0 / 606 / 569 / 72 | 0 / 1060 / 180 / 7 | 0 / 1 / 163 / 1083 | 0 / 636 / 360 / 136 | 0 / 940 / 170 / 22 | 0 / 10 / 53 / 1069 |
| median data σ, weak / middle / strong (mm) | 1.5e7 / 4.4 / 0.048 | ∞ / 3.6e6 / 0.049 | 0.73 / 0.095 / 0.010 | 1.2e11 / 31.6 / 0.037 | ∞ / 4.6e6 / 0.037 | 0.041 / 0.013 / 0.005 |
| \|cos\| of strongest / weakest direction to the source surface normal | 1.00 / 0.011 | 1.00 / 0.011 | 0.999 / 0.013 | 0.997 / 0.011 | 0.998 / 0.015 | 0.79 / 0.044 |
| true-motion energy in the recoverable subspace (median) | 0.16 | 0.017 | 1.0 | 0.015 | 0.00001 | 1.0 |
| exact-data linear oracle error, median (p90) | 9.1 (21.9) mm | 13.0 (22.2) mm | 0.008 (1.1) mm | 11.9 (14.2) mm | 14.1 (14.2) mm | 5e-5 (0.012) mm |

**Single pixel.**
- The strongest direction is the surface normal (|cos| ≈ 1), measured by depth.
- The weakest direction lies in the tangent plane (|cos| ≈ 0.01) and is unobservable.
- A second tangential direction is observable only where RGB texture or depth curvature exist: 51% of USB (641 / 1,247) and 44% of Blade (496 / 1,132) moving points have 2 or more recoverable directions.
- Blade's true motion is an in-plane slide, so only about 1.5% of its energy lies in the recoverable subspace.

**Patches.** Neighbouring measurements make all three directions informative for 87% (USB) and 94% (Blade) of points at k = 5. **This information is only as valid as the patch motion model** (§3).

## 3. Is the patch likelihood physically correct? (GT validation)

Deviation is the maximum over a patch of |true per-pixel image motion − modelled motion|. The material of each patch pixel is its GT first-hit owner on the same surface.

| frames 0→1 | USB k = 3 | USB k = 5 | USB k = 9 | Blade k = 3 | Blade k = 5 | Blade k = 9 |
|---|---|---|---|---|---|---|
| patches containing the other part (moving points) | 1.0% | 3.9% | 13.7% | 6.9% | 23.7% | 63.1% |
| rigid image patch: deviation median (p90), px | 0.046 (0.16) | 0.106 (0.32) | 0.29 (0.91) | 0.009 (1.36) | 0.043 (2.06) | 1.34 (2.06) |
| rigid image patch: patches with > 0.1 px deviation | 29% | 52% | 87% | 19% | 43% | 77% |
| 3D-translation patch (own measured depth): median (p90), px | 0.043 (0.15) | 0.094 (0.29) | 0.26 (0.94) | 0.005 (1.37) | 0.022 (2.05) | 1.36 (2.08) |
| 3D-translation patch, same-part pixels only: patches with > 0.1 px deviation | 26% | 46% | 79% | 16% | 39% | 75% |

Frames 1→2 are the same to within ±2 points (full tables).

**Likelihood behaviour near the truth.** Moving points, same-instant discrepancy added. On a curve, "argmin" is the position where the cost is lowest.

| model | true direction: argmin within ±0.1 of the truth | J(truth) < J(zero motion) | weak direction: argmin within ±0.1\|d\| of the truth | MAP from the truth: error median (p90) | MAP from the prediction: error median |
|---|---|---|---|---|---|
| USB rgbd-1 | 68% | 85% | 21% | 9.5 (22) mm | 13.9 mm |
| USB rgbd-3 | 70% | 83% | 22% | 8.0 (28) mm | 13.3 mm |
| USB rgbd-5 | 71% | 84% | 23% | 8.9 (35) mm | 14.3 mm |
| USB rgbd-9 | 69% | 85% | 27% | 4.7 (48) mm | 13.3 mm |
| USB depth-1 | 66% | 81% | 26% | 11.7 (22) mm | 15.1 mm |
| Blade rgbd-1 | 26% | 80% | 14% | 13.8 (14) mm | 14.2 mm |
| Blade rgbd-5 | 46% | 71% | 36% | 2.5 (14) mm | 14.2 mm |
| Blade rgbd-9 | 58% | 83% | 55% | 0.9 (5.8) mm | 14.2 mm |

The truth is a sharp minimum of the **median** objective along the true direction. The median curve decreases monotonically from zero motion to the truth, with no barrier on the straight path (figures). Per point, however:

- the minimum along the weakest information direction lies more than 0.1|d| (about 1.6 mm) from the truth for 73–79% of USB points under every model;
- the patch MAP started at the truth drifts 4.5–9 mm (median) with a heavier tail (p90 28–48 mm) than the single pixel (22 mm), because the rigid-patch residual at the truth is biased (§1);
- on Blade, large patches hold the truth (k = 9: 0.9 mm), but 63% of those patches straddle the static handle, so the likelihood is not physically valid.

From the prediction, no model reaches the truth. Larger patches also lose support when they move: 33% (k = 5) and 50% (k = 9) of USB points, and 77% and 98% of Blade points, lose at least one sample along the true-motion line. The identity-preserving acceptance therefore rejects those steps.

## 4. Silhouettes

Among GT-visible, same-surface moving identities, the true current position rounds to a background pixel in only 3 of 2,024 cases (USB) and 13 of 1,989 (Blade). All are 1 px from foreground, a bilinear neighbour is always foreground, and none has a valid material-sample support. For genuinely visible observations the nearest-pixel silhouette rule therefore loses at most about 0.7%.

The earlier 3,450 USB rejections ("background projection" of the GT state in the IEKF diagnostic) counted any camera of a point. That included cameras where the point is not GT-visible on its own surface, so it is a different population and not a loss of valid information.

## 5. Brightness / photometric mismatch (current image perturbed; MAP from the truth; static false motion)

| perturbation | USB rgbd-1 p90 / p99 | USB rgbd-5 p90 / p99 | USB gainnorm-5 p90 / p99 | USB gainnorm-9 p99 | USB depth-1 p99 | Blade rgbd-1 p99 |
|---|---|---|---|---|---|---|
| gain 1.02 | 4.2 / 25 mm | 10.5 / 67 mm | 3e-11 / 1.9 mm | 6.8 mm | 2e-10 mm | 4.0 mm |
| gain 1.08 | 4.1 / 54 mm | 16.8 / 84 mm | 4e-11 / 1.9 mm | 6.8 mm | 2e-10 mm | 12.5 mm |
| offset +0.02 | 2.8 / 35 mm | 12 / 75 mm | 0.55 / 7.5 mm | 17 mm | 2e-10 mm | 5.7 mm |

- Raw-RGB rows, single or patch, turn global photometric changes into false motion, and patches make it worse.
- Gain normalization removes gain sensitivity but not offset sensitivity.
- Depth is immune.

## 6. Answers

1. **Recoverable from these RGB-D images, per independent point:**
   - the normal component, from depth;
   - one tangential component where texture or depth curvature exist.
2. **Inherently unobservable from single-pixel material samples:**
   - the second tangential component;
   - in particular, Blade's in-plane slide.
3. **Do patches give a physically correct, better-conditioned likelihood?** Better conditioned: yes. Physically correct: no.
   - Neither the image-rigid nor the 3D-translation patch matches GT per-pixel motion. About half of USB's 5×5 patches deviate by more than 0.1 px (while the same-instant σ is about 0.1 px-equivalent), and Blade's patches straddle parts.
   - Their per-point minima are off the truth.
   - **Not implemented.**
4. **Noise.** An independently justified depth uncertainty is available without GT: the same-instant cross-camera discrepancy (1.1e-4 m USB, 6.9e-5 m Blade). It explains most of USB's temporal depth residual and corrects a 14–1000× underestimate. It does not by itself make the tangential motion observable.
5. **Photometric.** Raw-RGB temporal rows are not robust to global brightness changes. Any future RGB term should be gain/offset-invariant, or weighted by an independently calibrated photometric model.

**Possible next steps (your decision; none implemented):**
- **(a)** Adopt the same-instant cross-camera depth discrepancy as a declared, data-derived resampling term in material-following. This calibrates the covariance; it does not recover tangential motion.
- **(b)** Investigate a patch model whose per-pixel motion is validated. The GT results show the residual vanishes with true per-pixel motion, so the obstacle is modelling neighbour motion without coupling point states, which the current constraints forbid.
- **(c)** Accept per-point tangential unobservability and report it with honest covariance.
