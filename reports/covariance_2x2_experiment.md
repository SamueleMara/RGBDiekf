# 2×2 covariance experiment: measurement covariance R × velocity prior P_vv,0 (2026-10-29, HEAD e1e663b)

This follows `reports/static_point_equilibrium_root_cause_CURRENT.md`. It asks whether the static drift comes from the calibration of R, the calibration of P_vv,0, or their interaction.

Only the covariances change. The IEKF iteration, acceptance, visibility, masks and process noise q_c are untouched. The direction is forward and only the free stage runs.

## Arms and calibrations

**Arms:**

| arm | R | P_vv,0 | how it runs |
|---|---|---|---|
| A | current | current | the baseline of the audit |
| A′ (control) | current | current | through the `--sensor-noise config` route |
| B | calibrated | current | config route; only `rgb_model_variance` and `depth_model_variance` change |
| C | current | calibrated | `--velocity-prior-speed99 v99` (the existing flag) |
| D | calibrated | calibrated | B and C combined |

- **Current R:** the scene's measured temporal sensor noise, plus a model variance set by the *median* cross-camera moment match (`scene/noise_calibration.py`).
- **Current P_vv,0:** 1 (m/s)² I, the documented uncalibrated guess.
- **A′ is bit-identical to A in all 17 arrays, in all six scenes**, so the route itself changes nothing. Every difference between arms comes from the covariance.

**Calibrated R (your choice: "second moment at truth"):**
- **Rows:** every static point is evaluated at its static location, on 6–9 frames, with all other points at their GT location for visibility. A row is kept when its visibility is VALID, its 2×2 sample is valid, and the measured surface is not in front (the existing rule).
- **Rule:** one additive model variance m per block (RGB pooled, depth), solving mean(r² / (v_sensor + m)) = 1. Sensor variances are unchanged.
- **Static identity used:** GT for synthetic, by construction for the fixture, the depth-constant proxy for real scenes.
- **This is an oracle calibration:** it uses the static identity, so on real data it is only as good as the proxy.

**Calibrated P_vv,0 (your choice: "GT v99 per scene"):**
- **Rule:** v99 is the 99th percentile of the true speed of every point (static ones at 0) over every frame interval. P_vv = (v99 / 3.368)² I.
- **Not available for:** the real scenes (no GT speed, and no declared value was given), and the static fixture (GT v99 = 0, a degenerate prior). C and D are therefore run only on the three Sapien scenes.

**Calibrated values:**

| scene | σ_rgb current → calibrated | σ_D current → calibrated | at-truth tail beyond χ²₁,₉₉ (rgb / depth), current → calibrated | GT v99 → P_vv |
|---|---|---|---|---|
| storage | 0.0094 → 0.0203 | 5.7 µm → 0.78 mm | 7.7% / 3.0% → 0.9% / 1.0% | 0.532 m/s → 0.025 |
| USB | 0.0041 → 0.0339 | 55 µm → 0.69 mm | 11.1% / 10.8% → 0.9% / 2.1% | 1.330 m/s → 0.156 |
| stapler | 0.0051 → 0.0220 | 12 µm → 0.82 mm | 8.7% / 35.3% → 0.9% / 2.1% | 3.126 m/s → 0.861 |
| fixture | 0.0081 → 0.0089 | 17 µm (unchanged) | 4.1% / 0% → 3.0% / 0% | 0 (not run) |
| laptop (proxy) | 0.017 → 0.047 | 34 → 49 mm | 5.0% / 2.9% → 3.1% / 2.9% | n/a |
| black_box (proxy) | 0.028 → 0.052 | 20 → 9.5 mm | 4.6% / 0.1% → 1.8% / 4.4% | n/a |

Black_box is the one case where the second moment is *below* the current median-matched depth variance.

## Results

**Metrics:**
- Static points use the audit's static basis.
- "Step 10" is 0.17 s on synthetic and 0.4 s on real.
- The final frame is 59 for synthetic and the 2 s window end (frame 50) for real.
- "At truth" is the share of static points whose admitted (gated) or sampled rows exceed χ²₉₉ when evaluated at the static location; 1% is expected.
- The moving-part error is GT-based and the same for every arm (`tools/evaluate_projective_gt.py`, nearest-label moving points).

| scene | arm | σ_rgb / σ_D | P_vv (m²/s²) | frame-1 static \|δx\| p50 / p90 / p99 (mm) | frame-1 \|v\| p50 / p99 (m/s) | first injection \|δv\| p50 / p90 (m/s) | static error at step 10 p50 / > 10 cm | final static error p50 / p90 / > 10 cm | at truth: rows above χ²₉₉ (gated / sampled) | moving error final, median (true motion) | moving within 5 mm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| storage | A | 0.0094 / 0.00573 mm | 1 | 0.79 / 8.9 / 29.1 | 0.046 / 1.70 | 0.104 / 0.63 | 9.9 mm / 11.7% | 13.9 / 686 mm / 31.4% | 17.0% / 17.4% | 321 mm (532) | 6.1% |
| storage | B | 0.0203 / 0.779 mm | 1 | 0.66 / 6.4 / 21.6 | 0.038 / 1.26 | 0.077 / 0.45 | 8.0 mm / 7.5% | 10.8 / 488 mm / 27.1% | 3.2% / 3.4% | 210 mm (532) | 8.2% |
| storage | C | 0.0094 / 0.00573 mm | 0.0249 | 0.23 / 3.2 / 7.1 | 0.010 / 0.31 | 0.032 / 0.17 | 3.8 mm / 0.1% | 5.4 / 77 mm / 8.5% | 15.1% / 17.4% | 532 mm (532) | 2.2% |
| storage | D | 0.0203 / 0.779 mm | 0.0249 | 0.12 / 1.9 / 4.8 | 0.005 / 0.21 | 0.015 / 0.10 | 3.1 mm / 0.0% | 4.3 / 39 mm / 3.7% | 3.0% / 3.4% | 532 mm (532) | 7.4% |
| USB | A | 0.0041 / 0.0549 mm | 1 | 0.00 / 1.3 / 14.3 | 0.000 / 0.84 | 0.001 / 0.18 | 0.3 mm / 3.5% | 13.6 / 224 mm / 17.8% | 13.8% / 15.2% | 724 mm (899) | 0.0% |
| USB | B | 0.0339 / 0.692 mm | 1 | 0.00 / 0.6 / 9.5 | 0.000 / 0.56 | 0.000 / 0.09 | 0.0 mm / 1.9% | 3.0 / 139 mm / 12.7% | 3.5% / 3.5% | 642 mm (899) | 0.1% |
| USB | C | 0.0041 / 0.0549 mm | 0.156 | 0.00 / 1.0 / 9.2 | 0.000 / 0.52 | 0.000 / 0.13 | 0.0 mm / 1.5% | 5.5 / 134 mm / 12.1% | 13.5% / 15.2% | 713 mm (899) | 0.0% |
| USB | D | 0.0339 / 0.692 mm | 0.156 | 0.00 / 0.3 / 5.1 | 0.000 / 0.28 | 0.000 / 0.06 | 0.0 mm / 0.7% | 0.8 / 88 mm / 9.2% | 3.4% / 3.5% | 677 mm (899) | 0.0% |
| stapler | A | 0.0051 / 0.012 mm | 1 | 0.00 / 5.5 / 24.0 | 0.000 / 1.40 | 0.040 / 0.51 | 11.5 mm / 10.9% | 75.9 / 633 mm / 45.0% | 39.7% / 42.1% | 1634 mm (1659) | 0.0% |
| stapler | B | 0.0220 / 0.816 mm | 1 | 0.00 / 2.5 / 8.8 | 0.000 / 0.52 | 0.010 / 0.21 | 3.3 mm / 2.1% | 16.2 / 241 mm / 22.8% | 3.3% / 3.9% | 1609 mm (1659) | 0.0% |
| stapler | C | 0.0051 / 0.012 mm | 0.861 | 0.00 / 5.3 / 21.6 | 0.000 / 1.26 | 0.039 / 0.48 | 10.9 mm / 10.3% | 72.6 / 605 mm / 44.3% | 39.7% / 42.1% | 1632 mm (1659) | 0.0% |
| stapler | D | 0.0220 / 0.816 mm | 0.861 | 0.00 / 2.3 / 8.6 | 0.000 / 0.50 | 0.010 / 0.20 | 3.1 mm / 1.8% | 15.3 / 228 mm / 22.1% | 3.3% / 3.9% | 1610 mm (1659) | 0.0% |
| fixture | A | 0.0081 / 0.0172 mm | 1 | 0.16 / 1.1 / 3.4 | 0.005 / 0.10 | 0.010 / 0.04 | 0.3 mm / 0.0% | 0.3 / 2 mm / 0.0% | 3.9% / 4.0% | — (no GT) | — |
| fixture | B | 0.0089 / 0.0172 mm | 1 | 0.17 / 1.1 / 3.4 | 0.005 / 0.10 | 0.010 / 0.04 | 0.3 mm / 0.0% | 0.3 / 2 mm / 0.0% | 2.7% / 2.7% | — (no GT) | — |
| laptop (proxy, frames 0–50) | A | 0.0172 / 33.6 mm | 1 | 1.42 / 14.1 / 37.3 | 0.036 / 0.93 | 0.125 / 0.38 | 57.3 mm / 28.4% | 230.7 / 903 mm / 66.6% | 6.3% / 9.5% | — (no GT) | — |
| laptop (proxy, frames 0–50) | B | 0.0470 / 48.7 mm | 1 | 1.19 / 9.5 / 27.8 | 0.030 / 0.70 | 0.102 / 0.26 | 46.8 mm / 18.5% | 179.8 / 643 mm / 63.6% | 2.3% / 4.0% | — (no GT) | — |
| black_box (proxy, frames 0–50) | A | 0.0283 / 19.7 mm | 1 | 0.00 / 12.8 / 37.4 | 0.000 / 0.93 | 0.053 / 0.40 | 36.5 mm / 22.5% | 172.0 / 909 mm / 62.3% | 5.3% / 5.8% | — (no GT) | — |
| black_box (proxy, frames 0–50) | B | 0.0523 / 9.5 mm | 1 | 0.00 / 15.5 / 41.4 | 0.000 / 1.03 | 0.055 / 0.46 | 39.4 mm / 24.4% | 178.7 / 989 mm / 62.4% | 3.8% / 4.1% | — (no GT) | — |

## What it isolates

Below, "share" means the fraction of static points whose final error exceeds 10 cm, and each arm's ratio is relative to A.

**1. R calibration (B against A) removes the measurement inconsistency at the truth.**
- Points with a row beyond χ²₉₉ at their static location fall from 14–40% to 3.2–3.5% (synthetic) and from 5–6% to 2.3–3.8% (real).
- The effect on drift depends on the scene. The share falls in:
  - stapler: 45.0 → 22.8%, the largest depth bias;
  - USB: 17.8 → 12.7%;
  - storage: 31.4 → 27.1%;
  - laptop: 66.6 → 63.6%.
- On black_box it does not change (62.3 → 62.4%); its calibrated depth σ is smaller than the current one.

**2. P_vv calibration (C against A) acts through the size of the prior, in proportion to how far the GT v99 is below the current implied 3.37 m/s.**
- **Storage** (v99 0.53, P_vv 40× smaller): 31.4 → 8.5%. Frame-1 p99 speed 1.70 → 0.31 m/s.
- **USB** (v99 1.33): 17.8 → 12.1%.
- **Stapler** (v99 3.13, P_vv 0.86 ≈ 1): 45.0 → 44.3%. It is effectively the same prior, and nothing changes.
- **C leaves the at-truth inconsistency unchanged** (15–40%): R is still wrong. The rows are only gated slightly differently.

**3. Interaction (D).** Expressed as each arm's share relative to A:

| scene | B | C | D | B × C |
|---|---|---|---|---|
| storage | 0.86 | 0.27 | **0.12** | 0.23 |
| USB | 0.71 | 0.68 | 0.52 | 0.48 |
| stapler | 0.51 | 0.98 | 0.49 | 0.50 |

- The two effects are **close to independent and multiplicative**, with a positive interaction on storage.
- D gives the lowest static drift everywhere: storage 3.7%, USB 9.2%, stapler 22.1%.

**4. Neither calibration, nor both, removes the drift.**
- D still leaves 3.7–22% of static points more than 10 cm off.
- The first-frame displacement does not vanish (p99 4.8–8.6 mm), and the first injected velocity is still dt·L·δx.
- This residual is consistent with the confirmed numerical iteration instability (cell crossings, non-convergence, the corner knife edge). As noted in advance, covariance calibration does not address it.

**5. The moving parts are not tracked in any arm.**

| scene | A | B | C | D | true motion |
|---|---|---|---|---|---|
| storage | 321 mm | 210 mm | 532 mm | 532 mm | 532 mm |
| USB | 724 mm | 642 mm | 713 mm | 677 mm | 899 mm |
| stapler | 1634 mm | 1609 mm | 1632 mm | 1610 mm | 1659 mm |

- **Storage:** the calibrated prior makes moving tracking worse. The drawer moves at exactly v99 = 0.53 m/s, which is 3.4σ of the new prior. In C and D the moving points never move (error = true motion).
- **B** is the best arm for moving-part accuracy in all three Sapien scenes, but its improvements are small. At most 8% of moving points end within 5 mm (storage B).
- So a v99 taken from the 99th percentile of point speeds is too tight for a part that moves at constant speed. Lower static drift is being bought with lost tracking.

## Conclusion

**What explains the drift:**
- **Both calibrations matter, largely independently.** Which one dominates depends on the scene:
  - R where the depth model is strongly biased (stapler);
  - P_vv where the true speeds are far below the implied prior (storage);
  - comparable effects on USB.
- **Their combination is best for static points**, but leaves a residual (3.7–22%) attributable to the iteration instability.
- **Real scenes:** R calibration alone helps laptop little and black_box not at all. The P_vv arms cannot be run without a declared speed.

**Moving parts:** no arm restores moving-part accuracy. A calibration that only shrinks the velocity prior trades static stability for lost tracking.

**Decision needed (yours):**
- the velocity prior for real scenes (a declared v99);
- whether the oracle second-moment R should become a production calibration. It needs a static identity, which real scenes do not have.

## Reproduce

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf; PY=~/miniconda3/envs/rigeuler3dgs/bin/python
OUT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/static_equilibrium_audit_20261029_1718"
$PY tools/audit_static_equilibrium.py covariance-calibration "$OUT/$N/baseline" --out "$OUT/$N/arms"   # writes noise configs + train args
# arms: train.py <baseline args> --output "$OUT/$N/arms/<ARM>" + train_args_R_current | train_args_R_calibrated [+ train_args_Pvv_calibrated]
$PY tools/audit_static_equilibrium.py population "$OUT/$N/arms/<ARM>" --selection-from "$OUT/$N/baseline"
$PY tools/evaluate_projective_gt.py --run "$OUT/$N/arms/<ARM>" --scene "$SC" --out "$OUT/$N/arms/<ARM>/evaluation_gt"
$PY tools/audit_static_equilibrium.py arms A=... B=... C=... D=... --baseline "$OUT/$N/baseline" --out "$OUT/$N/arms/arms_summary.json"
```

**Run notes:**
- Arm A is the audit baseline. A′ (`arms/A_config`) reproduces it bit for bit.
- **Unintended C/D attempts:** my run script also tried C and D on the fixture and the real scenes, where no v99 exists. They failed at argument parsing, before any result was written. The leftover folders are renamed `D_INVALID_no_gt_v99_empty_args` and contain only the failure record.

**Tests:** `tests/test_static_equilibrium_audit.py` (10 tests, including the second-moment rule) passes.
