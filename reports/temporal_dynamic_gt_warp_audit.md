# GT-only audit of temporal RGB-D residuals at the true motion — USB_100109 and blade_103706, frames 0→1→2

**Scope.** Post-hoc, ground-truth-only falsification (`tools/audit_temporal_dynamic_correspondence.py`). Nothing here is read by any estimator. Run: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/material_warp_gt_audit/audit_20261102_104059_55e9f635` (`index.html`, per-scene `audit.json`, `traces.json`, `figures/`). Machine-readable summary: `reports/temporal_dynamic_gt_warp_audit.json`.

**Setup.**
- Full frame-0 endpoint catalogue: USB has 95,967 points, Blade has 34,369.
- True positions: x_k = T_GT(q_k − q_0) x_0 from the joint metadata.
- Sensor noise: the same `temporal_sensor_noise` call as `train.py`. It is quantization-level on this synthetic data:
  - RGB σ = 1.13e-3;
  - depth σ ≈ 8.3e-6·z on USB and 6.9e-8 m on Blade.
- R_warp = 0. No variance was guessed.

## 1. Ground-truth correspondence

| | USB | Blade |
|---|---|---|
| Source pixels reproject onto themselves (px / m) | p99 2.8e-14 / 4.4e-16 | 0 / 4.4e-16 |
| Non-source \|D_t(π(x_t))−Z_t\| vs same rows at τ (ratio) | 0.85, 0.90 | 0.35, 1.08 |
| Labelled part-cloud size (vertices) | 1,500 | 555 |
| NN label ≠ multi-frame on-surface vote | 1,922 + 3,113 points | 356 + 2,784 points |
| Label verified (NN and vote agree) | 39,798 moving / 47,588 static; 8,581 excluded | 2,782 moving / 21,546 static; 10,041 excluded |
| Verdict | VERIFIED | VERIFIED (for label-verified points) |

The nearest-neighbour label from the sparse labelled clouds is wrong near part contacts. Blade figure `blade_103706_point_7973.png` shows a yellow handle pixel labelled "moving" whose zero-motion residual is exactly 0. All class statistics below therefore use only points whose two independent labels agree.

## 2. Residuals at the true motion (moving, visible, same-surface rows; row-weighted camera medians)

Notation:
- A = old fixed-current-pixel Eulerian warp;
- B = material-following;
- Z = the same B rows at zero motion;
- C = depth at the true projection minus the true Z.

| | USB 0→1 | USB 1→2 | Blade 0→1 | Blade 1→2 |
|---|---|---|---|---|
| eligible / visible rows | 39,704 / 35,409 | 39,791 / 35,556 | 2,846 / 2,127 | 2,863 / 2,140 |
| occluded / reference pixel on another surface / support invalid | 107 / 167 / 4,132 | 225 / 205 / 4,048 | 0 / 96 / 607 | 12 / 80 / 600 |
| median source→current shift (px) | 1.76 | 1.77 | 1.38 | 1.38 |
| **B** \|r_D\| median (p90), m | **8.7e-5** (6.6e-4) | **9.2e-5** (6.9e-4) | 5.0e-5 (4.9e-3) | 3.3e-5 (4.7e-3) |
| B signed depth median, m | +6.5e-5 | +7.0e-5 | +8.0e-6 | +4.1e-6 |
| B \|r_RGB\| median | 4.1e-4 | 4.0e-4 | 2.7e-4 | 2.6e-4 |
| B NIS median (χ²₄ median 3.36) | 11.4 | 12.6 | 1.2e6 | 6.0e5 |
| **Z** (zero motion) \|r_D\| median | 4.0e-3 | 4.0e-3 | 1.7e-3 | 1.9e-3 |
| A (old warp) \|r_D\| median / NIS | 8.4e-5 / 9.5 | 7.2e-5 / 7.7 | 1.8e-4 / 1.0e7 | 3.8e-4 / 1.0e8 |
| non-source C at t / at τ (pure bilinear sampling error), m | 1.0e-4 / 1.1e-4 | 1.1e-4 / 1.2e-4 | — | — |

Static rows have exactly zero residual in every model (identical synthetic frames).

Per point, summed over visible rows:

| | USB 0→1 | USB 1→2 | Blade 0→1 | Blade 1→2 |
|---|---|---|---|---|
| truth preferred over zero motion | **78%** | **77%** | 46% | 51% |
| J per row at truth / at zero motion | 11.7 / 4,275 | 12.9 / 4,940 | 6.3e4 / 3.1e5 | 4.9e4 / 4.4e5 |
| z-score of true displacement, all channels / RGB only | 110 / 0.93 | 114 / 0.85 | 2,132 / 0.0 | 2,987 / 0.0 |

## 3. Gate A verdict

- **USB: PASS with qualification.**
  - The material-following residual at the true motion is not biased relative to the motion signal: depth 9e-5 m against 4e-3 m at zero motion, with a +6.5e-5 m signed bias.
  - Its size equals the bilinear sampling error of exactly on-surface points (C at τ ≈ C at t ≈ 1e-4 m). It is therefore an interpolation discrepancy, not a motion-model error.
  - Against the quantization-level declared noise it is overconfident by about ×2 in σ (NIS 11–13).
  - The old Eulerian warp is equally small at the truth on USB. Its pilot failure was the frozen current-pixel acceptance domain, not its residual.
- **Blade: PASS in physical units, but `IMAGE_MOTION_UNOBSERVABLE`.**
  - On 6 of 8 cameras the RGB-D at moving pixels is unchanged under zero motion: a textureless blade sliding in its own plane, so RGB-only z = 0.
  - Depth observes the motion only in cameras 2 and 6.
  - The 211–222 "textured interior" rows are the static handle contour occluding the sliding blade, not moving texture. The normalized statistics are meaningless at the declared depth σ of 6.9e-8 m (float quantization).

No `MATERIAL_WARP_INVALID_AT_GT`. Residual versus edge-distance and incidence plots, and 33 (USB) / 31 (Blade) traced points with pixel crops are in the HTML. Of the requested IDs:
- USB 87290 and 85696 are verified moving and traced.
- USB 67827 and Blade 27016 are **static by both independent labels** (votes static:moving 6:4 and 13:4), so they are traced as static references, not moving points.
